echo "installing pytest..."
pip install pytest
echo "installing package locally"
python3 setup.py develop
echo "running tests.."
pytest -v tests/test.py  
echo "tests complete!"