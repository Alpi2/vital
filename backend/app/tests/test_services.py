def test_analyzer_stub():
    from app.services.ecg_analyzer import analyze_signal
    res = analyze_signal([0.0, 1.0, -0.5])
    assert 'bpm' in res
