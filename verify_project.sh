#!/bin/bash
# Project verification script

echo "================================"
echo "Project Verification"
echo "================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
echo ""

# Check directory structure
echo "2. Checking directory structure..."
required_dirs=("src" "scripts" "configs" "tests" "models" "checkpoints" "results")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ (missing)"
    fi
done
echo ""

# Check required files
echo "3. Checking required files..."
required_files=(
    "README.md"
    "LICENSE"
    "requirements.txt"
    "pyproject.toml"
    ".gitignore"
    "configs/default.yaml"
    "configs/ablation.yaml"
    "scripts/train.py"
    "scripts/evaluate.py"
    "scripts/predict.py"
)
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
    fi
done
echo ""

# Check Python syntax
echo "4. Checking Python syntax..."
python3 -m py_compile scripts/train.py 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ scripts/train.py"
else
    echo "  ✗ scripts/train.py (syntax error)"
fi

python3 -m py_compile scripts/evaluate.py 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ scripts/evaluate.py"
else
    echo "  ✗ scripts/evaluate.py (syntax error)"
fi

python3 -m py_compile scripts/predict.py 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ scripts/predict.py"
else
    echo "  ✗ scripts/predict.py (syntax error)"
fi
echo ""

# Check YAML configs
echo "5. Checking YAML configs..."
python3 -c "import yaml; yaml.safe_load(open('configs/default.yaml'))" 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ configs/default.yaml"
else
    echo "  ✗ configs/default.yaml (invalid)"
fi

python3 -c "import yaml; yaml.safe_load(open('configs/ablation.yaml'))" 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ configs/ablation.yaml"
else
    echo "  ✗ configs/ablation.yaml (invalid)"
fi
echo ""

# Count lines of code
echo "6. Project statistics..."
echo "  Python files: $(find . -name '*.py' -type f | wc -l)"
echo "  Total Python LOC: $(find . -name '*.py' -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')"
echo "  Test files: $(find tests -name 'test_*.py' -type f | wc -l)"
echo ""

echo "================================"
echo "Verification complete!"
echo "================================"
echo ""
echo "To run the project:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Train model: python scripts/train.py --config configs/default.yaml"
echo "  3. Evaluate: python scripts/evaluate.py --checkpoint models/best_preference_model.pt"
echo "  4. Generate: python scripts/predict.py --checkpoint models/best_preference_model.pt --prompt 'your text'"
