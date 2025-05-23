cd ..
pip install -r requirements.txt
cd trl
pip install -e .

pip install flash-attn --no-build-isolation

cd ../open-r1
pip install -e ".[dev]"
