from models.automata.sax import SAXTransformer


def test_sax_returns_symbol_per_value():
    transformer = SAXTransformer(alphabet_size=3)

    result = transformer.transform([-1.0, 0.0, 1.0])

    assert len(result) == 3
    assert all(symbol in {"a", "b", "c"} for symbol in result)
