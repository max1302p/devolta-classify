"""Response shape, ordering, rounding, batch order."""

from tests.conftest import AUTH

LABELS = ["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"]
TEXT = "Mein Kassensystem druckt seit gestern keine Bons mehr"
MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def test_response_schema(client):
    response = client.post(
        "/classify", json={"text": TEXT, "labels": LABELS}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "label",
        "score",
        "results",
        "multi_label",
        "model",
        "duration_ms",
    }
    assert body["multi_label"] is False
    assert body["model"] == MODEL_ID
    assert isinstance(body["duration_ms"], int) and body["duration_ms"] >= 0
    assert [item["label"] for item in body["results"]] != []
    assert all(set(item) == {"label", "score"} for item in body["results"])
    assert {item["label"] for item in body["results"]} == set(LABELS)


def test_results_sorted_descending_and_top_matches(client):
    body = client.post(
        "/classify", json={"text": TEXT, "labels": LABELS}, headers=AUTH
    ).json()

    scores = [item["score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["label"] == body["results"][0]["label"]
    assert body["score"] == body["results"][0]["score"]
    # FakePipeline weights ascending -> the last label wins once sorted.
    assert body["label"] == LABELS[-1]


def test_scores_rounded_to_four_decimals(client):
    body = client.post(
        "/classify", json={"text": TEXT, "labels": ["a", "b", "c"]}, headers=AUTH
    ).json()

    for item in body["results"]:
        assert round(item["score"], 4) == item["score"]
    assert body["results"][-1]["score"] == 0.1667  # 1/6 rounded


def test_default_hypothesis_template_is_used(client, fake_pipe):
    client.post("/classify", json={"text": TEXT, "labels": LABELS}, headers=AUTH)
    assert fake_pipe.calls[-1]["hypothesis_template"] == "Diese Nachricht betrifft {}."


def test_custom_hypothesis_template_is_passed_through(client, fake_pipe):
    client.post(
        "/classify",
        json={
            "text": TEXT,
            "labels": LABELS,
            "hypothesis_template": "In diesem Text geht es um {}.",
        },
        headers=AUTH,
    )
    assert fake_pipe.calls[-1]["hypothesis_template"] == "In diesem Text geht es um {}."


def test_multi_label_is_passed_through_and_echoed(client, fake_pipe):
    body = client.post(
        "/classify",
        json={"text": TEXT, "labels": LABELS, "multi_label": True},
        headers=AUTH,
    ).json()

    assert fake_pipe.calls[-1]["multi_label"] is True
    assert body["multi_label"] is True


def test_labels_are_stripped_before_inference(client, fake_pipe):
    client.post(
        "/classify",
        json={"text": TEXT, "labels": ["  Rechnung ", "Spam"]},
        headers=AUTH,
    )
    assert fake_pipe.calls[-1]["candidate_labels"] == ["Rechnung", "Spam"]


def test_batch_keeps_order_and_schema(client):
    items = [
        {"text": "Erster Text", "labels": ["A", "B"]},
        {"text": "Zweiter Text", "labels": ["C", "D", "E"], "multi_label": True},
        {"text": "Dritter Text", "labels": ["F", "G"]},
    ]
    response = client.post("/classify/batch", json={"items": items}, headers=AUTH)
    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 3
    assert [{item["label"] for item in r["results"]} for r in results] == [
        {"A", "B"},
        {"C", "D", "E"},
        {"F", "G"},
    ]
    assert [r["multi_label"] for r in results] == [False, True, False]
    assert [r["label"] for r in results] == ["B", "E", "G"]


def test_batch_sends_texts_in_order(client, fake_pipe):
    items = [{"text": f"Text {i}", "labels": ["A", "B"]} for i in range(5)]
    client.post("/classify/batch", json={"items": items}, headers=AUTH)

    assert [call["text"] for call in fake_pipe.calls] == [f"Text {i}" for i in range(5)]


def test_text_is_never_logged(client, caplog):
    secret = "strictly confidential customer text"
    with caplog.at_level("INFO"):
        client.post("/classify", json={"text": secret, "labels": LABELS}, headers=AUTH)

    assert any(record.message == "classify" for record in caplog.records)
    assert secret not in caplog.text
    record = next(r for r in caplog.records if r.message == "classify")
    assert record.n_labels == len(LABELS)
    assert record.top_label == LABELS[-1]
    assert isinstance(record.duration_ms, int)
