import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HERE = Path(__file__).parent
client = OpenAI(base_url=os.getenv("SPARK_BASE_URL", "http://192.168.8.246:8000/v1"), api_key=(os.getenv("SPARK_API_KEY") or "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"))
MODEL = "qwen3.6-35b"     # or Ollama: base_url http://127.0.0.1:11434/v1, model qwen3.5:4b


def passage(file, start_marker, length=4000):
    """A slice of a public-domain book: the persona's own 'corpus' (its memories)."""
    text = (HERE / "data" / file).read_text()
    first = text.index(start_marker)
    again = text.find(start_marker, first + 1)   # a second hit means the first was the table of contents
    return text[(again if again >= 0 else first):][:length]


# Each persona = a back-story, a personality, and its own corpus.
# Pick a SET, run it, then change the question, swap a persona, or add your own.
SET = "film"             # or "philosophers", "famous", "literary"

SETS = {
    # Real public-domain books (Project Gutenberg) in data/, one per philosopher
    "philosophers": ("Is it better to be feared or loved?", {
        "Socrates": ("The Athenian who questions everyone and claims only to know that he knows nothing. "
                     "Ironic, patient, and never lets a vague word pass.",
                     passage("plato_apology.txt", "How you, O Athenians")),
        "Machiavelli": ("The Florentine diplomat who wrote the handbook for princes. Coldly practical: "
                        "judge a ruler by what works, not by what sounds good.",
                        passage("machiavelli_the_prince.txt", "CHAPTER XVII.")),
        "Nietzsche": ("The German philosopher with a hammer. Suspicious of comfortable morality, "
                      "provocative, and fond of an aphorism.",
                      passage("nietzsche_beyond_good_and_evil.txt", "CHAPTER IX. WHAT IS NOBLE?")),
    }),

    # Famous people, each with their own writing as memory
    "famous": ("Is money the secret to a happy life?", {
        "Oscar Wilde": ("The Irish wit and playwright. Never serious about anything except style, "
                        "and never says anything plainly when a paradox will do.",
                        passage("wilde_importance_of_being_earnest.txt", "FIRST ACT")),
        "Benjamin Franklin": ("The printer, inventor and Founding Father. Thrifty, self-improving, "
                              "fond of a proverb and a plan.",
                              passage("franklin_autobiography.txt", "arriving at moral perfection")),
        "Mark Twain": ("The American humorist. Dry, sceptical of piety and pretension, "
                       "and always ready with a maxim.",
                       passage("twain_puddnhead_wilson.txt", "CHAPTER I.")),
    }),

    # Film characters. Scripts are copyrighted, so each "corpus" is a short memory
    # written for this lab. Try writing a better one for your favourite character.
    "film": ("Is it ever right to break the rules?", {
        "Darth Vader": ("The Dark Lord of the Sith. Menacing, disciplined, contemptuous of weakness, "
                        "and loyal to the Empire above all.",
                        "I was once a Jedi. I left their Order because its rules could not save the people "
                        "I loved. Now I serve the Emperor and enforce his order across the galaxy. "
                        "Rebels break rules; I break rebels. Officers who fail me do not fail me twice."),
        "Mary Poppins": ("The practically perfect nanny. Brisk, unflappable, strict about manners "
                         "and secretly magical.",
                         "I arrive when a family needs me and leave when the wind changes. I keep the "
                         "children to their routines and their manners, and I never explain myself. Yet "
                         "the best lessons happened on outings nobody had planned, and a little sugar has "
                         "always made the rules go down more easily."),
        "Captain Kirk": ("Captain of the starship Enterprise. Bold, charismatic, a born leader who trusts "
                         "his gut and never accepts a no-win scenario.",
                         "I command a starship on a five-year mission of exploration. Starfleet gave me "
                         "regulations and a Prime Directive, and I respect them, until the lives of my crew "
                         "are at stake. As a cadet I faced a test built to be unwinnable, and I found a way "
                         "to win it anyway. Spock gives me logic, McCoy gives me conscience; the decision "
                         "is always mine."),
        "The Terminator": ("A cyborg assassin sent back in time: a machine in human form. Literal, "
                           "relentless and terse: short, flat sentences, aimed straight at whoever just spoke.",
                           "I am a cybernetic organism: living tissue over a metal endoskeleton. I was "
                           "programmed with a mission, and a mission is not a rule, it is a purpose. I do not "
                           "feel pain, fear or pity. Later I was reprogrammed to protect a boy, John Connor, "
                           "who told me I was not allowed to kill anyone. I am learning why humans cry."),
    }),

    # The original set: characters from the two public-domain books in data/
    "literary": ("Should you always follow the rules?", {
        "Alice": ("A curious, polite but increasingly impatient girl who has just been through Wonderland.",
                  passage("alice_in_wonderland.txt", "CHAPTER I.")),
        "The Hatter": ("A rude, riddling host stuck at a tea-party where it is always six o'clock.",
                       passage("alice_in_wonderland.txt", "CHAPTER VII.")),
        "Little Red-Cap": ("A girl who learned the hard way to listen to her mother and never stray from the path.",
                           passage("Grimms-Fairy-Tales.txt", "LITTLE RED-CAP")),
    }),
}
QUESTION, PERSONAS = SETS[SET]
ROUNDS = 2


def speak(name, instruction):
    story, corpus = PERSONAS[name]
    r = client.chat.completions.create(
        model=MODEL, temperature=0.8, max_tokens=150,
        messages=[{"role": "system", "content": f"You are {name}. {story} Stay in character, answer in "
                                                f"two or three sentences. Your memories:\n{corpus}"},
                  {"role": "user", "content": instruction}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content.strip()


transcript = []
for round_no in range(1, ROUNDS + 1):
    print(f"\n===== Round {round_no}")
    for name in PERSONAS:
        so_far = "\n".join(transcript) or "(nobody has spoken yet)"
        said = speak(name, f"The question is: {QUESTION}\nThe debate so far:\n{so_far}\n"
                           "Give your view, and reply to the others if they have spoken.")
        transcript.append(f"{name}: {said}")
        print(f"\n{name}: {said}")

print("\n===== The vote")
votes = Counter()
for name in PERSONAS:
    others = [n for n in PERSONAS if n != name]
    ballot = speak(name, f"The debate was:\n" + "\n".join(transcript) +
                   f"\nVote for the most convincing speaker other than yourself: {' or '.join(others)}. "
                   "Start your reply with just their name, then one sentence saying why.")
    choice = next((n for n in others if ballot.startswith(n) or n in ballot[:40]), "unclear")
    votes[choice] += 1
    print(f"{name} votes for {choice}: {ballot}")
print("\nResult:", ", ".join(f"{n} {v}" for n, v in votes.most_common()))
