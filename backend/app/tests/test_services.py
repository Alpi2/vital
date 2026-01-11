def test_analyzer_stub():
    from app.services.ecg_analyzer import analyze_signal
    res = analyze_signal([0.0, 1.0, -0.5])
    assert 'bpm' in res


def test_analyze_signal_return_structure():
    from app.services.ecg_analyzer import analyze_signal
    samples = [0.0] * 100
    res = analyze_signal(samples)
    assert isinstance(res, dict)
    assert 'bpm' in res
    assert isinstance(res['bpm'], int)
