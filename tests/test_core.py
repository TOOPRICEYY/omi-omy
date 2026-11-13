import numpy as np

from attrib.align import assign_words
from attrib.diarize import Cluster, overlap_ratio
from attrib.embed import l2norm
from attrib.identify import Gallery, Identifier, IdentifyConfig, UnknownTracker
from attrib.metrics import der, false_accept_rate, identification_error, word_speaker_error
from attrib.transcribe import parse_response, split_points
from attrib.types import UNKNOWN, Region, Turn, Word
from attrib.vad import EnergyGate, merge_regions, pack, split_conversations

SR = 16000


def test_pack_and_timemap_roundtrip():
    x = np.arange(SR * 10, dtype=np.float32)
    regions = [Region(1.0, 2.0), Region(5.0, 6.5)]
    packed, tm = pack(x, regions, spacer=0.5)
    assert len(packed) == int(1.0 * SR) + int(0.5 * SR) + int(1.5 * SR)
    assert tm.to_orig(0.25) == 1.25
    assert abs(tm.to_orig(1.5 + 0.75) - 5.75) < 1e-9
    # A time inside the spacer clamps to the end of the previous region.
    assert tm.to_orig(1.2) == 2.0


def test_merge_and_split():
    rs = merge_regions([Region(0, 1), Region(1.5, 2), Region(10, 11)], max_gap=1.0)
    assert [(r.start, r.end) for r in rs] == [(0, 2), (10, 11)]
    convs = split_conversations([Region(0, 1), Region(50, 51), Region(400, 401)], gap=120)
    assert [len(c) for c in convs] == [2, 1]


def test_energy_gate_keeps_onsets():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR * 6).astype(np.float32) * 0.001
    x[2 * SR: 3 * SR] += np.sin(np.arange(SR) / SR * 2 * np.pi * 220).astype(np.float32) * 0.2
    regions = EnergyGate(preroll=0.5, hangover=1.0)(x)
    assert len(regions) == 1
    assert regions[0].start <= 1.55 and regions[0].end >= 3.9


def test_assign_words_by_overlap_and_snap():
    turns = [Turn(0, 2, "A"), Turn(1.8, 4, "B")]
    ws = [Word("hi", 0.1, 0.4), Word("over", 1.7, 2.3), Word("late", 4.2, 4.4), Word("far", 9, 9.2)]
    out = assign_words(ws, turns)
    assert [w.speaker for w in out] == ["A", "B", "B", None]


def test_overlap_ratio():
    assert overlap_ratio([Turn(0, 10, "A"), Turn(5, 10, "B")]) == 0.5


def test_der_perfect_and_permuted():
    ref = [Turn(0, 5, "A"), Turn(5, 10, "B")]
    hyp = [Turn(0, 5, "x"), Turn(5, 10, "y")]
    assert der(ref, hyp)["total"] == 0.0
    # Identification needs the actual names.
    assert identification_error(ref, hyp)["total"] == 1.0
    assert identification_error(ref, [Turn(0, 5, "A"), Turn(5, 10, UNKNOWN)])["total"] == 0.5


def test_der_confusion_and_miss():
    ref = [Turn(0, 10, "A")]
    r = der(ref, [Turn(0, 5, "x")])
    assert abs(r["miss"] - 0.5) < 0.01 and r["confusion"] == 0


def test_wder_and_far():
    ref = [Word("a", 0, 1, "A"), Word("b", 1, 2, "B")]
    hyp = [Word("a", 0, 1, "A"), Word("b", 1, 2, "A")]
    assert word_speaker_error(ref, hyp) == 0.5
    assert false_accept_rate([Turn(0, 10, "stranger")], [Turn(0, 5, "A")], {"A"}) == 0.5


def _cluster(label, v, loud=-30.0):
    return Cluster(label, [Turn(0, 1, label)], l2norm(v), [l2norm(v)], loud, 1.0)


def _world(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    people = {p: l2norm(rng.standard_normal(dim)) for p in ["wearer", "ann", "bob"]}
    g = Gallery()
    for p, v in people.items():
        for _ in range(5):
            g.add(p, v + 0.1 * rng.standard_normal(dim))
    cohort = rng.standard_normal((200, dim))
    return rng, people, g, cohort


def test_identify_known_and_unknown():
    rng, people, g, cohort = _world()
    ident = Identifier(g, cohort, IdentifyConfig(wearer=None))
    stranger = l2norm(rng.standard_normal(32))
    m = ident.identify([_cluster("S0", people["ann"]), _cluster("S1", stranger)])
    assert m["S0"].person == "ann" and m["S0"].confidence > 0.5
    assert m["S1"].person == UNKNOWN


def test_identify_one_person_per_cluster_and_merge():
    rng, people, g, cohort = _world()
    ident = Identifier(g, cohort, IdentifyConfig(wearer=None))
    a1 = people["bob"] + 0.05 * rng.standard_normal(32)
    a2 = people["bob"] + 0.05 * rng.standard_normal(32)
    m = ident.identify([_cluster("S0", a1), _cluster("S1", a2)])
    # A split cluster that clearly matches bob is merged into bob.
    assert m["S0"].person == "bob" and m["S1"].person == "bob"


def test_split_voice_does_not_get_the_next_best_name():
    rng, people, g, cohort = _world()
    # "ann" sounds a little like bob here, enough to clear the threshold as a second choice.
    g.people["ann"] = [l2norm(people["ann"] + 0.6 * people["bob"] + 0.05 * rng.standard_normal(32)) for _ in range(5)]
    ident = Identifier(g, cohort, IdentifyConfig(wearer=None))
    b1 = people["bob"] + 0.05 * rng.standard_normal(32)
    b2 = people["bob"] + 0.05 * rng.standard_normal(32)
    assert ident.score(l2norm(b2), "ann") >= ident.cfg.threshold                 # would pass as ann
    m = ident.identify([_cluster("S0", b1), _cluster("S1", b2)])
    assert m["S0"].person == "bob" and m["S1"].person == "bob"                   # both bob, not bob and ann
    # A real ann next to bob is still ann.
    m = ident.identify([_cluster("S0", b1), _cluster("S1", g.people["ann"][0])])
    assert m["S0"].person == "bob" and m["S1"].person == "ann"


def test_raw_floor_stops_an_outlier_sample_from_matching_everyone():
    rng, people, g, cohort = _world(dim=64)
    # An entry learned from distant speech: mostly "channel", a direction no cohort voice has.
    cohort[:, 0] = 0.0
    channel = np.eye(64)[0]
    far = lambda v: l2norm(0.7 * l2norm(v) + channel + 0.05 * rng.standard_normal(64))
    g.people["bob"] = [far(people["bob"]) for _ in range(5)]
    stranger = far(rng.standard_normal(64))                                      # another distant voice
    assert Identifier(g, cohort, IdentifyConfig(wearer=None)).identify([_cluster("S0", stranger)])["S0"].person == "bob"
    ident = Identifier(g, cohort, IdentifyConfig(wearer=None, raw_floor=0.75))
    assert ident.raw(stranger, "bob") < 0.75 < ident.raw(far(people["bob"]), "bob")
    assert ident.identify([_cluster("S0", stranger)])["S0"].person == UNKNOWN
    assert ident.identify([_cluster("S0", far(people["bob"]))])["S0"].person == "bob"
    assert ident.identify([_cluster("S0", people["ann"])])["S0"].person == "ann"  # others unaffected


def test_priors_break_ties():
    rng, people, g, cohort = _world()
    mix = l2norm(people["ann"] + people["bob"])
    ident = Identifier(g, cohort, IdentifyConfig(wearer=None, threshold=-10))
    assert ident.identify([_cluster("S0", mix)], priors={"ann": 0.95, "bob": 0.05})["S0"].person == "ann"
    assert ident.identify([_cluster("S0", mix)], priors={"ann": 0.05, "bob": 0.95})["S0"].person == "bob"


def test_gallery_stays_diverse_and_bounded():
    g = Gallery(max_per_person=3)
    base = np.eye(4)[0]
    for v in [base, base, base * 0.99 + 0.01, np.eye(4)[1]]:
        g.add("p", v)
    m = g.matrix("p")
    assert len(m) == 3
    assert (m @ np.eye(4)[1]).max() > 0.99  # the distinct sample survived


def test_unknown_tracker_recurring():
    tr = UnknownTracker(threshold=0.9)
    v = l2norm(np.ones(8))
    for cid in ["c1", "c2", "c3"]:
        tr.observe(v, cid)
    tr.observe(l2norm(np.arange(8.0) - 3.5), "c4")
    assert tr.recurring(3) == [0]


def test_transcript_parse_and_split():
    body = {"words": [{"word": "hello", "start": 1.0, "end": 1.4}, {"word": "there", "start": 1.4, "end": 1.8}]}
    assert [(w.text, w.start) for w in parse_response(body, offset=10.0)] == [("hello", 11.0), ("there", 11.4)]
    ws = parse_response({"segments": [{"start": 3.0, "end": 4.0, "text": "no word times"}]})
    assert ws[0].text == "no" and abs(ws[2].end - 4.0) < 1e-9
    x = np.ones(16000 * 50, np.float32)
    x[16000 * 18: 16000 * 19] = 0  # a quiet second to cut at
    cuts = split_points(x, max_s=20, search_s=5)
    assert cuts[0] == 0 and cuts[-1] == len(x)
    assert 16000 * 18 <= cuts[1] <= 16000 * 19
    assert all(b - a <= 20 * 16000 for a, b in zip(cuts, cuts[1:]))
