import megadescribe as md
import pytest

def test_typeerror_at_initiation():
    with pytest.raises(TypeError):
        md.ColumnClassifier(0)