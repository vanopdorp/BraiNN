import random
import multiprocessing as mp
import json

random.seed(42)

SUBJECTS = [
    "the dog", "the cat", "the bird", "the fish", "the boy", "the girl",
    "the man", "the woman", "the baby", "the mouse", "the cow", "the horse"
]

VERB_OBJECTS = {
    "runs": ["in the park", "on the street", "in the garden"],
    "walks": ["in the park", "on the street", "in the house"],
    "sleeps": ["in the bed", "on the couch", "in the house"],
    "sits": ["on the chair", "on the couch", "on the bed"],
    "stands": ["in the garden", "on the street", "by the tree"],
    "flies": ["in the sky", "over the park"],
    "swims": ["in the river", "in the lake", "in the water"],
    "jumps": ["over the fence", "over the log"],
    "eats": ["the food", "the grass", "the meal"],
    "drinks": ["the water", "the milk"],
    "plays": ["with the ball", "in the garden"],
    "sings": ["a song"],
    "cries": ["softly"],
    "laughs": ["loudly"]
}

ADVERBS = [
    "quickly", "slowly", "calmly", "quietly", "softly", "happily"
]

PREPOSITIONS = [
    "in", "on", "under", "behind", "next to"
]

LOCATIONS = [
    "the house", "the garden", "the park", "the street", "the river"
]

CONJ_CAUSE = ["because", "since"]
CONJ_CONTRAST = ["but"]
CONJ_WHILE = ["while"]

EMOTIONS = [
    "is scared", "is happy", "is angry", "is tired", "is sad", "is nervous"
]

WEATHER = [
    "it is raining", "the sun is shining", "it is snowing", "the wind is blowing hard"
]

def gen_phase1():
    out = []
    for s in SUBJECTS:
        for v in VERB_OBJECTS:
            out.append(f"{s} {v}")
    return out

def gen_phase2():
    out = []
    for s in SUBJECTS:
        for v in VERB_OBJECTS:
            for adv in ADVERBS:
                out.append(f"{s} {v} {adv}")
    return out

def gen_phase3():
    out = []
    for s in SUBJECTS:
        for v, objs in VERB_OBJECTS.items():
            for obj in objs:
                out.append(f"{s} {v} {obj}")
    return out

def gen_phase4():
    out = []
    for s in SUBJECTS:
        for v, objs in VERB_OBJECTS.items():
            for obj in objs:
                for conj in CONJ_CAUSE:
                    for emo in EMOTIONS:
                        out.append(f"{s} {v} {obj} {conj} {s} {emo}")
    for s in SUBJECTS:
        for v1 in VERB_OBJECTS:
            for adv1 in ADVERBS:
                for conj in CONJ_CONTRAST:
                    for v2 in VERB_OBJECTS:
                        for adv2 in ADVERBS:
                            out.append(f"{s} {v1} {adv1} {conj} {s} {v2} {adv2}")
    for s in SUBJECTS:
        for v, objs in VERB_OBJECTS.items():
            for obj in objs:
                for conj in CONJ_WHILE:
                    for w in WEATHER:
                        out.append(f"{s} {v} {obj} {conj} {w}")
    return out

def gen_phase5():
    out = []
    for s1 in SUBJECTS:
        for v1, objs1 in VERB_OBJECTS.items():
            for obj1 in objs1:
                for s2 in SUBJECTS:
                    for v2, objs2 in VERB_OBJECTS.items():
                        for obj2 in objs2:
                            out.append(f"{s1} {v1} {obj1}. {s2} {v2} {obj2}")
    return out

def run_generator(gen_func, n_samples=None):
    data = gen_func()
    random.shuffle(data)
    if n_samples is not None:
        data = data[:n_samples]
    return data

def generate_curriculum_json(
    n_phase1=5000,
    n_phase2=5000,
    n_phase3=10000,
    n_phase4=10000,
    n_phase5=5000,
    n_workers=5,
    output_file="curriculum.json"
):
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(
            run_generator,
            [
                (gen_phase1, n_phase1),
                (gen_phase2, n_phase2),
                (gen_phase3, n_phase3),
                (gen_phase4, n_phase4),
                (gen_phase5, n_phase5),
            ]
        )

    curriculum = {
        "phase1_simple_svo": results[0],
        "phase2_svo_adv": results[1],
        "phase3_svo_prep_loc": results[2],
        "phase4_compound": results[3],
        "phase5_stories": results[4],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(curriculum, f, indent=2, ensure_ascii=False)

    print(f"JSON saved as: {output_file}")

if __name__ == "__main__":
    generate_curriculum_json()
