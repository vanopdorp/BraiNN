import random
import multiprocessing as mp
import json

random.seed(42)

SUBJECTS = [
    "the dog", "the cat", "the bird", "the fish", "the boy", "the girl",
    "the man", "the woman", "the baby", "the mouse", "the cow", "the horse"
]

VERB_OBJECTS = {
    "runs": ["in the park", "on the street", "in the garden", "towards the house", "around the tree", "through the field"],
    "walks": ["in the park", "on the street", "in the house", "along the river", "through the forest", "towards the shop"],
    "jumps": ["over the fence", "over the log", "over the rock", "across the stream", "over the puddle"],
    "climbs": ["the tree", "the hill", "the ladder", "the stairs", "the rock"],
    "crawls": ["under the table", "through the grass", "under the bed", "through the tunnel"],

    "sits": ["on the chair", "on the couch", "on the bed", "on the grass", "on the rock"],
    "stands": ["in the garden", "on the street", "by the tree", "near the river", "in the doorway"],
    "lies": ["on the bed", "on the floor", "in the grass", "on the couch"],

    "barks": ["loudly", "softly", "at the bird", "at the stranger"],
    "meows": ["softly", "loudly", "at the window"],
    "flies": ["in the sky", "over the park", "over the river", "around the tree"],
    "swims": ["in the river", "in the lake", "in the water", "across the pond"],

    "eats": ["the food", "the grass", "the meal", "the apple", "the bread", "the cookie"],
    "drinks": ["the water", "the milk", "the juice", "from the bowl"],
    "chews": ["the toy", "the stick", "the bone", "the leaf"],

    "sings": ["a song", "softly", "happily"],
    "cries": ["softly", "loudly", "in the room"],
    "laughs": ["loudly", "happily", "quietly"],

    "plays": ["with the ball", "in the garden", "with the toy", "with the stick", "with the water"],
    "talks": ["to the girl", "to the boy", "to the woman", "to the man"],
    "helps": ["the boy", "the girl", "the woman", "the man"],
    "hugs": ["the dog", "the cat", "the baby", "the friend"],
    "watches": ["the bird", "the sky", "the river", "the clouds", "the trees"],
    "follows": ["the dog", "the cat", "the boy", "the girl"],

    "thinks about": ["the game", "the food", "the walk", "the story"],
    "dreams about": ["the sky", "the adventure", "the forest", "the river"],

    "looks at": ["the tree", "the house", "the garden", "the river", "the sky"],
    "points at": ["the bird", "the car", "the house", "the flower"],

    "waits": ["in the rain", "under the tree", "in the sunshine", "by the door"],
    "rests": ["in the shade", "under the tree", "on the grass"],

    "opens": ["the door", "the window", "the box"],
    "closes": ["the door", "the window", "the book"],
    "carries": ["the bag", "the box", "the toy", "the basket"],
    "pushes": ["the cart", "the door", "the box"],
    "pulls": ["the rope", "the cart", "the toy"],

    "finds": ["a stone", "a leaf", "a stick", "a flower"],
    "picks up": ["the toy", "the stick", "the ball", "the book"],
    "drops": ["the toy", "the stick", "the ball", "the book"],

    "feels": ["happy", "sad", "tired", "excited", "calm"],
    "seems": ["happy", "angry", "calm", "nervous"],

    "waits for": ["the bus", "the friend", "the dog"],
    "looks for": ["the toy", "the ball", "the cat", "the dog"],

    "remembers": ["the game", "the story", "the walk"],
    "forgets": ["the toy", "the plan", "the idea"],
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
    n_phase3=5000,
    n_phase4=5000,
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
