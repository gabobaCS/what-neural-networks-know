from nltk.corpus import wordnet as wn
from typing import List, Optional, Tuple

def synset_id_to_lemmas(synset_id: str) -> str:
    """
    Convert an ImageNet-style synset ID (e.g., 'n01440764')
    to all lemma names joined as a single string.
    """
    offset = int(synset_id[1:])   # drop leading 'n'
    pos = synset_id[0]            # 'n' = noun
    synset = wn.synset_from_pos_and_offset(pos, offset)
    return ", ".join(lemma.name() for lemma in synset.lemmas())


def synset_id_to_gloss(synset_id: str) -> str:
    """
    Return the WordNet gloss (definition) for a given synset id, e.g. 'n01440764'.
    """
    s = wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
    return s.definition()

def synset_id_to_lemmas_and_gloss(synset_id: str) -> str:
    """
    Convert a WordNet-style synset ID (e.g., 'n01440764') to a string
    with lemma names and its gloss (definition).
    
    Example:
      'n01440764' -> 'tench, Tinca_tinca — freshwater dace-like game fish...'
    """
    offset = int(synset_id[1:])      # drop leading 'n'
    pos = synset_id[0]               # 'n' = noun
    synset = wn.synset_from_pos_and_offset(pos, offset)
    lemmas = ", ".join(lemma.name() for lemma in synset.lemmas())
    gloss = synset.definition()
    return f"{lemmas} — {gloss}"

def synset_id_to_lexname(synset_id: str) -> str:
    """
    Return the WordNet lexname (semantic category) for a given synset id, e.g. 'n01440764'.
    """
    s = wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
    return s.lexname()

def synset_id_to_coarse_labels(synset_id: str, depth: int) -> set[str]:
    """
    Given a synset ID (e.g., 'n02084071') and a depth, return all possible
    coarse labels at that depth across all hypernym paths.
    """
    try:
        pos = synset_id[0]            # 'n'
        offset = int(synset_id[1:])   # e.g., 2084071
        synset = wn.synset_from_pos_and_offset(pos, offset)

        coarse_labels = {
            path[depth].name()
            for path in synset.hypernym_paths()
            if len(path) > depth
        }
        return coarse_labels

    except Exception:
        return set()
    
def find_minimum_path_between_synsets(synset_id1: str, synset_id2: str) -> Optional[List[str]]:
    """
    Find the minimum path between two synset IDs (e.g., 'n01440764', 'n02084071').
    Returns a list of synset names representing the shortest path, or None if no path exists.
    """
    # Convert synset IDs to synset objects
    s1 = wn.synset_from_pos_and_offset(synset_id1[0], int(synset_id1[1:]))
    s2 = wn.synset_from_pos_and_offset(synset_id2[0], int(synset_id2[1:]))
    
    # Find the shortest path
    path = s1.shortest_path_distance(s2, simulate_root=True)
    
    # If no path exists, return None
    if path is None:
        return None
    
    # Get the actual path of synsets
    path_synsets = s1.hypernym_paths()[0] if s1.hypernym_paths() else [s1]
    target_paths = s2.hypernym_paths()[0] if s2.hypernym_paths() else [s2]
    
    # Find common hypernym (lowest common subsumer)
    lcs = s1.lowest_common_hypernyms(s2)
    if not lcs:
        return None
    
    common_synset = lcs[0]
    
    # Build path from s1 to common ancestor
    path_to_common = []
    current = s1
    while current != common_synset:
        path_to_common.append(current.name())
        hypernyms = current.hypernyms()
        if not hypernyms:
            break
        current = hypernyms[0]  # Take first hypernym
    path_to_common.append(common_synset.name())
    
    # Build path from common ancestor to s2
    path_from_common = []
    current = s2
    while current != common_synset:
        path_from_common.append(current.name())
        hypernyms = current.hypernyms()
        if not hypernyms:
            break
        current = hypernyms[0]  # Take first hypernym
    
    # Combine paths (reverse the second path and remove duplicate common synset)
    path_from_common.reverse()
    full_path = path_to_common + path_from_common
    
    return full_path

def get_minimum_path_distance(synset_id1: str, synset_id2: str) -> Optional[int]:
    """
    Get the minimum path distance between two synset IDs.
    Returns the distance as an integer, or None if no path exists.
    """
    s1 = wn.synset_from_pos_and_offset(synset_id1[0], int(synset_id1[1:]))
    s2 = wn.synset_from_pos_and_offset(synset_id2[0], int(synset_id2[1:]))
    
    return s1.shortest_path_distance(s2, simulate_root=True)

def get_path_with_lemmas(synset_id1: str, synset_id2: str) -> Optional[List[Tuple[str, str]]]:
    """
    Get the minimum path between two synsets with both synset names and lemmas.
    Returns a list of tuples (synset_name, lemmas_string), or None if no path exists.
    """
    path = find_minimum_path_between_synsets(synset_id1, synset_id2)
    if path is None:
        return None
    
    result = []
    for synset_name in path:
        synset = wn.synset(synset_name)
        lemmas = ", ".join(lemma.name() for lemma in synset.lemmas())
        result.append((synset_name, lemmas))
    
    return result
