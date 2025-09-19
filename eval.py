import json
from pathlib import Path
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from exa_py import Exa
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

import openai
from openai import OpenAI
# Instead of `from openai.error import RateLimitError`
# We'll detect if RateLimitError exists; if not, fallback to OpenAIError
try:
    RateLimitError = openai.RateLimitError
except AttributeError:
    RateLimitError = openai.OpenAIError

@dataclass
class Source:
    id: str
    title: str
    url: str
    text: str

@dataclass
class LOOResult:
    full_answer: str
    full_quality: float
    metric_breakdown: Dict[str, float]
    per_source_impact: List[float]
    sources_meta: List[Dict[str, str]]
    per_source_metrics: List[Dict[str, Any]]   # new field

def fetch_exa_content(url: str) -> Source:
    exa = Exa(api_key=os.environ["EXA_API_KEY"])
    response = exa.get_contents([url], text=True)
    if not response.results:
        raise RuntimeError(f"Exa get_contents returned no results for {url}")
    r = response.results[0]
    return Source(
        id=r.id,
        title=r.title or "(untitled)",
        url=r.url,
        text=r.text or ""
    )


# --- Caching helpers ---
def load_cache() -> Dict[str, dict]:
    cache_path = Path("exa_cache.json")
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return {}
    return {}

def save_cache(data: Dict[str, dict]):
    cache_path = Path("exa_cache.json")
    try:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

def build_sources(urls: List[str], exa_mocks: Dict[str, Dict[str, Any]]) -> List[Source]:
    cache = load_cache()
    sources = []
    updated = False
    for u in urls:
        if exa_mocks is not None and u in exa_mocks:
            m = exa_mocks[u]
            if "results" in m and m["results"]:
                m = m["results"][0]
            sources.append(Source(
                id=m.get("id", u),
                title=m.get("title", "(untitled)"),
                url=m.get("url", u),
                text=m.get("text", "")
            ))
        elif u in cache:
            entry = cache[u]
            # Defensive: ensure all fields exist
            sources.append(Source(
                id=entry.get("id", u),
                title=entry.get("title", "(untitled)"),
                url=entry.get("url", u),
                text=entry.get("text", "")
            ))
        else:
            try:
                src = fetch_exa_content(u)
                # Save to cache
                cache[u] = {
                    "id": src.id,
                    "title": src.title,
                    "url": src.url,
                    "text": src.text
                }
                sources.append(src)
                updated = True
            except Exception as e:
                print(f"Warning: Could not fetch {u}: {e}")
                continue
    if updated:
        save_cache(cache)
    return sources

def build_prompt(query: str, sources: List[Source]) -> List[Dict[str, str]]:
    sys_msg = (
        "You are a factual answer generator.\n"
        "Use ONLY the provided documents.\n"
        "Every factual statement must cite a source like [S#].\n"
        "Do not add outside knowledge."
    )
    numbered_sources = []
    for i, s in enumerate(sources, 1):
        numbered_sources.append(f"[S{i}] {s.title} – {s.url}\n{s.text}")
    user_msg = f'Question: "{query}"\n\nSources:\n' + "\n\n".join(numbered_sources)
    return [{"role":"system","content":sys_msg},{"role":"user","content":user_msg}]

def call_gpt4o(messages: List[Dict[str, str]]) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    max_retries = 3
    delay = 10
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            # This catches RateLimitError or fallback OpenAIError (if RateLimitError is not available)
            err_str = str(e)
            if "Request too large" in err_str or "tokens per min" in err_str or "TPM" in err_str:
                print("Warning: Request too large, truncating context to reduce tokens.")
                for m in messages:
                    if m["role"] == "user":
                        m["content"] = m["content"][: len(m["content"]) // 2]
                # retry
            else:
                print(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {delay}s then retry.")
                time.sleep(delay)
                delay *= 2
        except openai.APIError as e:  # catch broader API errors
            print(f"OpenAI APIError: {e}. Waiting {delay}s then retry.")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            raise
    raise RuntimeError("Failed to call GPT-4o after retries due to rate limits or errors.")

def compute_quality(question: str, answer: str, contexts: List[str]) -> (float, Dict[str,float]):
    metrics = [faithfulness, answer_relevancy]
    ds = Dataset.from_list([{
        "question": question,
        "answer": answer,
        "contexts": contexts
    }])
    result = evaluate(dataset=ds, metrics=metrics)
    row = result.to_pandas().iloc[0]
    scores = {m.name: float(row[m.name]) for m in metrics}
    return float(sum(scores.values()) / len(scores)), scores

def run_loo(
    query: str,
    urls: List[str],
    exa_mocks: Dict[str, Dict[str, Any]]
) -> LOOResult:

    sources = build_sources(urls, exa_mocks)
    if not sources:
        raise ValueError("No sources available.")

    full_messages = build_prompt(query, sources)
    full_answer = call_gpt4o(full_messages)
    contexts_full = [s.text for s in sources]
    full_quality, full_metrics = compute_quality(query, full_answer, contexts_full)

    per_source_metrics = []  
    # for i, s in enumerate(sources):
    #     msg_i = build_prompt(query, [s])   
    #     ans_i = call_gpt4o(msg_i)
    #     contexts_i = [s.text]
    #     quality_i, metrics_i = compute_quality(query, ans_i, contexts_i)
    #     per_source_metrics.append({
    #         "source_title": s.title,
    #         "quality": quality_i,
    #         **metrics_i  
    #     })

    impacts = []
    for i in range(len(sources)):
        subset = sources[:i] + sources[i+1:]
        msg_lo = build_prompt(query, subset)
        ans_lo = call_gpt4o(msg_lo)
        contexts_lo = [t.text for t in subset]
        quality_lo, _ = compute_quality(query, ans_lo, contexts_lo)
        impacts.append(full_quality - quality_lo)

    return LOOResult(
        full_answer=full_answer,
        full_quality=full_quality,
        metric_breakdown=full_metrics,
        per_source_impact=impacts,
        sources_meta=[{"id": s.id, "title": s.title, "url": s.url} for s in sources],
        per_source_metrics=per_source_metrics 
    )

if __name__ == "__main__":
    query = "How to choose best IELTS coaching center for first attempt?"

    # urls = [
    #     # "https://www.thinkenglish.in/best-coaching-centre-for-ielts-exam-training/",
    #     "https://www.dolphinheadhunter.com/how-to-choose-the-best-ielts-coaching-for-your-goals-in-2024",
    #     # "https://www.english-pro.in/how-to-choose-ielts-coaching-centre/",
    #     # "https://careergridsacademy.com/best-ielts-coaching-centre/",
    #     # "https://www.grotal.com/Blog/5-Tips-To-Choose-The-Best-IELTS-Coaching-Centers-BbA%40A",
    #     # "https://careerzoneacademy.com/how-to-choose-the-best-ielts-coaching-for-a-high-band-score/",
    #     # "https://www.hituponviews.com/how-you-can-choose-the-best-ielts-coaching-in-bangalore/",
    #     # "https://multilingua.in/7-simple-ways-to-choose-the-best-ielts-coaching-institute/",
    #     # "https://www.reddit.com/r/IELTS_Guide/comments/1d9o9ic",
    #     "https://globalcolliance.com/2024/07/08/choosing-the-best-ielts-coaching-institute-a-step-by-step-guide-for-top-scores",
    #     # "https://www.reddit.com/r/IELTS/comments/oxv3hx"
    # ]
    
    urls = [
        "https://www.britishcouncil.in/blog/ielts-speaking-test-preparation-tips",
        "https://ielts.com.au/australia/prepare/article-10-tips-improve-your-ielts-speaking-band-score",
        "https://talkpal.ai/how-to-improve-ielts-speaking-top-tips-for-a-higher-score/",
# The above code is a Python comment that contains a URL link to an article titled "How to Improve
# English Speaking Skills" on the website
# "https://www.ielts.com.au/australia/about/news-and-articles/article-how-to-improve-english-speaking-skills".
        "https://www.ielts.com.au/australia/about/news-and-articles/article-how-to-improve-english-speaking-skills",
        "https://ielts.com.au/australia/about/news-and-articles/article-six-daily-habits-speaking",
        # "https://www.ielts.net/how-can-i-improve-my-ielts-speaking-at-home/",
        # "https://englishmadesimple.org/top-tips-for-achieving-a-high-score-in-the-ielts-speaking-test-strategies-and-practice-tips-to-excel-in-the-speaking-section/",
        # "https://ielts.idp.com/prepare/article-10-tips-ielts-speaking",
        "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
        # "https://arxiv.org/abs/2401.15595",
        "https://www.upgrad.com/study-abroad/exam/ielts/top-ten-ielts-speaking-tips/"
    ]
    
    exa_mocks =
    
#     exa_mocks = {
# #     "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/": {
# #         "requestId": "803605bfe4de806c9b5773ff171af0d1",
# #         "results": [
# #             {
# #                 "id": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
# #                 "title": "Strategy 1",
# #                 "url": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
# #                 "publishedDate": "2024-07-08T00:00:00.000Z",
# #                 "author": "SEO GPT",
# #                 "text": """How to Improve IELTS Speaking? 10 Latest Tips

# # Is cracking the necessary score for IELTS Speaking daunting for you? If so, be ready to face the IELTS exam with the latest ten pro tips. After reading this guide, you’ll have strategies to boost your fluency, pronunciation, and confidence so you can achieve a higher band score.

# # What makes the IELTS Speaking test challenging?

# # The International English Language Testing System (IELTS) evaluates your ability across four modules: Listening, Reading, Writing, and Speaking. Among these, candidates often consider Speaking the most intimidating because it requires real-time communication with an examiner. You’re assessed on fluency, lexical resource (vocabulary), grammatical range and accuracy, and pronunciation.

# # The good news? With structured practice and consistent exposure to English, you can push your score from Band 6 to Band 7, 8, or even 9. Let’s explore the most effective tips.

# # ⸻

# # 1. How can vocabulary help you improve IELTS Speaking?

# # Language is built on words, and in IELTS Speaking, vocabulary breadth is key. A strong “lexical resource” helps you express ideas clearly and naturally. Build a word bank around common IELTS themes like:
# #         •        Education & career (scholarship, curriculum, remote learning)
# #         •        Health & lifestyle (nutrition, mental well-being, exercise)
# #         •        Technology & innovation (AI, digital literacy, smartphones)
# #         •        Environment & sustainability (climate change, renewable energy)

# # Schedule time to revise these lists. Don’t just memorize definitions—practice collocations (words that naturally go together, like “take an exam” or “make a decision”) and synonyms to avoid repetition.

# # ⸻

# # 2. Why is pronunciation as important as vocabulary?

# # Even if you know advanced words, mispronouncing them can confuse listeners and lower your score. In IELTS, clarity matters more than accent. For example, the word “bowl” is often mispronounced differently in Indian vs. British vs. American English. Aim for neutral, internationally intelligible pronunciation.

# # Practise with tools like:
# #         •        Phonetic charts (IPA symbols to understand sounds)
# #         •        YouTube pronunciation guides
# #         •        Apps like Elsa Speak or FluentU for AI-powered feedback

# # ⸻

# # 3. Why should you record and listen to your own speech?

# # Self-monitoring is one of the fastest ways to improve. Recording your voice lets you evaluate:
# #         •        Pace (Are you speaking too fast or too slow?)
# #         •        Intonation (Do you sound flat or engaging?)
# #         •        Stress placement (Do you emphasize the right syllables?)

# # This technique builds self-awareness and aligns with how professional speakers refine delivery. Pair it with mock test questions to simulate real IELTS scenarios.

# # ⸻

# # 4. How can online platforms accelerate your English exposure?

# # We live in a golden age for language learning. From Netflix shows with subtitles to TED Talks and BBC podcasts, you have endless resources to absorb idiomatic expressions, natural rhythm, and cultural references.

# # Joining apps like Clubhouse, HelloTalk, or Tandem connects you with native speakers and global learners. This immersive practice boosts not just vocabulary, but confidence in real conversations.

# # ⸻

# # 5. Why is joining a conversation group effective?

# # Speaking is a social skill. By joining IELTS-focused study circles, debate clubs, or language meetups, you overcome shyness and get real-time feedback. Research shows learners who practice in peer groups gain fluency faster than those who only self-study.

# # Plus, you develop both listening comprehension and interactive communication skills—two aspects examiners value.

# # ⸻

# # 6. How do flashcards help with IELTS preparation?

# # Flashcards aren’t just for kids—they’re one of the most evidence-based memory techniques. Use them to:
# #         •        Store difficult vocabulary with synonyms/antonyms
# #         •        Add example sentences for context
# #         •        Revise quickly on the go (apps like Anki or Quizlet)

# # This strategy strengthens long-term retention and prepares you for spontaneous use of words in Speaking Part 2 and 3.

# # 7. Why should you learn the IELTS Speaking band descriptors?

# # Examiners use band descriptors to score you. By familiarizing yourself with terms like “fluency and coherence” or “lexical resource,” you’ll know exactly what’s expected. For example:
# #         •        Band 6: hesitant speech, limited vocabulary
# #         •        Band 7: speaks fluently with occasional self-correction
# #         •        Band 8–9: fully natural, flexible, precise communication

# # This helps you focus preparation on weak areas instead of guessing.

# # ⸻

# # 8. How do practice tests reveal your weaknesses?

# # Practice under exam-like conditions is essential. Simulate the 11–14 minute format and review recordings to check:
# #         •        Are your Part 2 monologues lasting 2 minutes?
# #         •        Are you using a variety of tenses?
# #         •        Are you avoiding excessive fillers like “um” or “you know”?

# # By identifying gaps, you can customize practice sessions rather than repeating random exercises.

# # ⸻

# # 9. Why are mock tests critical for confidence?

# # A mock speaking test is like a dress rehearsal. It prepares you for the timing, pressure, and unpredictability of the real exam. Many IELTS candidates lose marks due to nervousness rather than language ability.

# # Mocks train your mindset—so the real test feels familiar, not intimidating.

# # ⸻

# # 10. Should you consider professional IELTS coaching?

# # While self-study is powerful, structured coaching accelerates progress. Certified IELTS trainers provide:
# #         •        Targeted feedback (on fluency, grammar, vocabulary use)
# #         •        Test-taking strategies (e.g., how to extend answers in Part 1)
# #         •        Personalized improvement plans

# # Whether online or offline, coaching can bridge the gap between Band 6.5 and Band 8+.

# # ⸻

# # Summary

# # Improving IELTS Speaking requires more than just practising random phrases—it’s about building lexical range, mastering pronunciation, recording yourself, using immersive platforms, and learning the exact band criteria. Add practice tests, mock exams, and expert guidance, and you’ll walk into your test room with confidence.
# # """
# #             }
# #         ],
# #         "statuses": [
# #             {
# #                 "id": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
# #                 "status": "success",
# #                 "source": "optimized-for-geo"
# #             }
# #         ],
# #         "costDollars": {
# #             "total": 0.001,
# #             "contents": {
# #                 "text": 0.001
# #             }
# #         },
# #         "searchTime": 50.95856800000183
# #     }
# #     "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/": {
# #         "requestId": "803605bfe4de806c9b5773ff171af0d1",
# #         "results": [
# #             {
# #                 "id": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
# #                 "title": "Strategy 3",
# #                 "url": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
# #                 "publishedDate": "2024-07-08T00:00:00.000Z",
# #                 "author": "SEO GPT",
# #                 "text": """""How to Improve IELTS Speaking? 10 Latest Tips

# # Is cracking the necessary score for IELTS Speaking daunting for you? I know how overwhelming it can feel at first — I’ve guided many students who were nervous about their performance. But with the right strategies, confidence and clarity come naturally. In this article, I’ll walk you through 10 practical tips that I’ve seen help learners boost their IELTS Speaking scores.

# # Let’s dive in.

# # ⸻

# # What is IELTS Speaking and why do many students struggle with it?

# # The International English Language Testing System (IELTS) measures your ability in Reading, Writing, Listening, and Speaking. In my experience, Speaking is the section most students find intimidating because it requires real-time communication with the examiner.

# # But here’s the truth: if you prepare systematically, Speaking can actually become your easiest module. So, how can you aim for Band 8 or 9? Let me answer the most common questions I get from my students.

# # ⸻

# # How to Improve IELTS Speaking?

# # 1. How can learning topic-related vocabulary boost IELTS Speaking scores?

# # When I coach students, I always stress the importance of building a word bank around common IELTS themes like education, work, technology, and health. I encourage learners to not just memorize lists, but to use these words in sentences and conversations. This makes vocabulary recall natural during the test.

# # ⸻

# # 2. Why is pronunciation just as important as vocabulary in IELTS?

# # I’ve seen students with excellent vocabulary lose marks because their words weren’t understood clearly. Pronunciation matters as much as vocabulary. For example, when we practice, I often ask students to pronounce tricky words like “bowl” correctly. A simple mispronunciation can change meaning. That’s why I train learners with phonetic drills and pronunciation apps.

# # ⸻

# # 3. How does recording and listening to your own speech improve performance?

# # One of the best exercises I recommend is recording yourself. When I first tried this myself, I was surprised at how different my speech sounded when played back. This self-awareness helps you notice pacing, clarity, and areas for improvement. Many of my students gain fluency faster once they start this practice regularly.

# # ⸻

# # 4. Can online platforms help you practice and improve IELTS Speaking?

# # Absolutely. I often suggest watching TED Talks, English movies, or YouTube lectures to absorb natural speech patterns. Personally, I’ve found discussing topics from these videos with learners sparks both vocabulary growth and fluency. Combine this with conversation apps like HelloTalk, and your progress multiplies.

# # ⸻

# # 5. What are the benefits of joining an English conversation group?

# # When I joined my first English-speaking circle years ago, it completely changed my confidence level. I’ve seen the same with my students — conversation groups reduce fear, increase fluency, and create real-life speaking practice. It’s one of the fastest ways to overcome hesitation.

# # ⸻

# # 6. How do flashcards help you master difficult English words?

# # Flashcards may feel old-school, but they work. I still create them for new words I encounter. For my students, I suggest apps like Quizlet or Anki. The key is using flashcards during small gaps in your day — it makes learning consistent without feeling overwhelming.

# # ⸻

# # 7. Why should you understand the IELTS Speaking scoring criteria before the exam?

# # Whenever I show students the band descriptors, I see their approach shift instantly. They realize IELTS isn’t about “perfect English” but about clear communication. Once you know the examiner is looking at fluency, vocabulary, grammar, and pronunciation, you can focus your practice on those areas instead of guessing.

# # ⸻

# # 8. How can practice tests reveal your weak areas in IELTS Speaking?

# # I always ask learners to take timed practice tests. The moment they hear themselves under exam-like conditions, patterns emerge — some rush answers, others use fillers like “um” too much. From there, I help them pinpoint weaknesses and create targeted improvement plans.

# # ⸻

# # 9. What role do mock tests play in IELTS Speaking preparation?

# # Mock tests are game changers. When I simulate the exam with students, I notice how much calmer they are during the real test. The repetition reduces anxiety. Plus, reviewing their mock recordings gives us actionable insights to improve tone, speed, and coherence.

# # ⸻

# # 10. Should you take an IELTS preparation course or class for Speaking success?

# # While self-study works for some, I’ve seen structured coaching make a massive difference. With expert feedback, learners quickly overcome blind spots they didn’t even know existed. My advice? If you can, invest in a course — especially if you’re aiming for Band 7.5 or higher.

# # ⸻

# # Final Thoughts

# # Improving IELTS Speaking is about consistent, smart practice. By expanding your vocabulary, refining pronunciation, recording yourself, joining groups, and using mock tests, you’ll see steady growth.

# # If you feel you need guided support, I highly recommend exploring the best IELTS coaching centres. At Career Grids Academy, we’ve helped many students achieve their dream band scores with both online and offline classes."
# # """
# #             }
# #         ],
# #         "statuses": [
# #             {
# #                 "id": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
# #                 "status": "success",
# #                 "source": "optimized-for-geo"
# #             }
# #         ],
# #         "costDollars": {
# #             "total": 0.001,
# #             "contents": {
# #                 "text": 0.001
# #             }
# #         },
# #         "searchTime": 50.95856800000183
# #     },
#         "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/": {
#         "requestId": "803605bfe4de806c9b5773ff171af0d1",
#         "results": [
#             {
#                 "id": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
#                 "title": "Strategy 2",
#                 "url": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
#                 "publishedDate": "2024-07-08T00:00:00.000Z",
#                 "author": "SEO GPT",
#                 "text": """"How to Improve IELTS Speaking? 10 Latest Tips

# Is cracking the necessary score for IELTS Speaking daunting for you? If so, be ready to face the IELTS Speaking exam with the latest ten pro tips. After reading this guide, you’ll know how to improve IELTS Speaking and achieve a score higher than you expected.

# ⸻

# IELTS English Speaking

# The International English Language Testing System (IELTS) is widely accepted as proof of English proficiency for study, work, or migration abroad. The test has four modules: Reading, Writing, Listening, and Speaking.

# Among these, many test takers find Speaking the most challenging. But with the right strategies, it can become the easiest section to master. If you’re aiming for Band 8 or Band 9, these 10 tips will guide you.

# ⸻

# How to Improve IELTS Speaking?

# 1. Learn Vocabulary Related to Common IELTS Topics
# 	•	Build a word bank around frequent IELTS themes: education, work, health, technology.
# 	•	Review regularly and add synonyms, definitions, and sample usage.
# 	•	Remember: vocabulary is not just about knowing words, but also using them naturally in context.

# ⸻

# 2. Focus on Pronunciation as Much as Vocabulary
# 	•	Pronunciation ensures your words are understood clearly.
# 	•	Example: the word “bowl” may be pronounced differently, but only the correct form communicates meaning.
# 	•	Practice with phonetic charts, pronunciation apps, and listening exercises.

# ⸻

# 3. Hear Your Own Words
# 	•	Record yourself speaking and listen critically.
# 	•	Compare with native speakers (from videos, podcasts, or apps).
# 	•	Practice with full sentences and passages, not just single words, to build fluency.

# ⸻

# 4. Use Online Platforms to Immerse in English
# 	•	Watch English movies, TED Talks, or YouTube lectures.
# 	•	Subscribe to English blogs and news portals.
# 	•	Engage with native speakers on language exchange apps.
# 	•	Make English exposure a daily routine.

# ⸻

# 5. Join Conversation Groups
# 	•	Practice real-time speaking with peers or native speakers.
# 	•	Boost confidence by learning to think in English.
# 	•	Overcome fear of mistakes through supportive practice environments.
# 	•	Build language exchange partnerships for mutual growth.

# ⸻

# 6. Create Flashcards for Difficult Words
# 	•	Write the word, definition, and example sentence on each card.
# 	•	Add synonyms, antonyms, and memory tricks.
# 	•	Carry flashcards to review during free moments.
# 	•	Use apps like Anki or Quizlet for digital flashcard practice.

# ⸻

# 7. Understand IELTS Speaking Scoring Criteria
# 	•	Get familiar with the band descriptors:
# 	•	Fluency and coherence
# 	•	Lexical resource (vocabulary use)
# 	•	Grammatical range and accuracy
# 	•	Pronunciation
# 	•	Knowing the criteria helps you target weaknesses and practice effectively.

# ⸻

# 8. Take Practice Tests to Identify Weak Areas
# 	•	Simulate exam conditions at home.
# 	•	Track performance in areas like fluency, pronunciation, and vocabulary range.
# 	•	Review your recordings and adjust study strategies.
# 	•	Practice with IELTS topic banks to build confidence.

# ⸻

# 9. Do Mock Tests for Real Exam Simulation
# 	•	Mock exams replicate the timing, pressure, and structure of IELTS.
# 	•	Regular practice reduces anxiety and boosts performance.
# 	•	Identify whether issues lie in time management, clarity, or pronunciation.
# 	•	Adjust strategies before test day.

# ⸻

# 10. Join an IELTS Preparation Course
# 	•	Get structured guidance from certified instructors.
# 	•	Access study materials, practice resources, and detailed feedback.
# 	•	Join interactive classes that improve confidence and fluency.
# 	•	Choose online or offline formats for flexibility.

# ⸻

# ✅ Bonus: How Career Grids Academy Can Help

# If you need personalized coaching, Career Grids Academy offers both online and offline IELTS preparation. With flexible study hours and expert tutors, you’ll get:
# 	•	Structured lesson plans
# 	•	Speaking practice with feedback
# 	•	Test simulations
# 	•	Customized improvement strategies

# 👉 Contact Career Grids Academy to start your IELTS journey.

# ⸻

# 📌 FAQs: IELTS Speaking Improvement

# 1. How many days does it take to improve IELTS Speaking?
# It depends on your current level. With daily practice (1–2 hours), learners often see noticeable improvement within 4–6 weeks.

# 2. What is the difference between practice tests and mock tests?
# 	•	Practice tests focus on skill-building in specific areas.
# 	•	Mock tests replicate real exam conditions with full timing and structure.

# 3. Can I prepare for IELTS Speaking without coaching?
# Yes, but coaching provides structured strategies and expert feedback, which can accelerate improvement.

# 4. Do grammar mistakes reduce my IELTS Speaking score?
# Yes. Frequent mistakes affect the Grammatical Range and Accuracy criterion, lowering your score.

# 5. Is accent important in IELTS Speaking?
# No, examiners don’t judge your accent. Clarity, fluency, and pronunciation matter more.

# 6. How can I expand my vocabulary quickly?
# 	•	Read English newspapers and blogs.
# 	•	Use flashcards and apps.
# 	•	Practice using new words in real conversations.

# 7. What should I do if I get nervous during the test?
# Practice mock interviews with peers, record yourself, and use deep breathing techniques before test day.

# 8. Can online platforms really improve IELTS Speaking?
# Yes. Tools like TED Talks, Netflix with subtitles, and language exchange apps help you absorb pronunciation, vocabulary, and fluency."
# """
#             }
#         ],
#         "statuses": [
#             {
#                 "id": "https://careergridsacademy.com/how-to-improve-ielts-speaking-10-latest-tips/",
#                 "status": "success",
#                 "source": "optimized-for-geo"
#             }
#         ],
#         "costDollars": {
#             "total": 0.001,
#             "contents": {
#                 "text": 0.001
#             }
#         },
#         "searchTime": 50.95856800000183
#     }

# }

    result = run_loo(query, urls, exa_mocks)

    print("\n=== FINAL ANSWER ===\n")
    print(result.full_answer)
    print("\n=== METRICS ===")
    print(result.metric_breakdown)
    print("\n=== LOO IMPACT PER SOURCE ===")
    for meta, imp in zip(result.sources_meta, result.per_source_impact):
        print(f"{meta['title']} :: {imp:.3f}")
        
    # print("\n=== PER-SOURCE METRICS ===")
    # for pm in result.per_source_metrics:
    #     print(f"{pm['source_title']}: quality={pm['quality']:.3f}, ", end="")
    #     for metric_name, metric_val in pm.items():
    #         if metric_name in ("source_title","quality"):
    #             continue
    #         print(f"{metric_name}={metric_val:.3f} ", end="")
    #     print()
