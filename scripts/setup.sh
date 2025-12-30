#!/bin/bash
# rlbook.ai Development Setup Script
#
# This script sets up the complete development environment for rlbook.ai,
# including both the Node.js frontend and Python RL code package.
#
# Usage:
#   ./scripts/setup.sh          # Full setup
#   ./scripts/setup.sh --node   # Node.js only
#   ./scripts/setup.sh --python # Python only

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
SETUP_NODE=true
SETUP_PYTHON=true

if [[ "$1" == "--node" ]]; then
    SETUP_PYTHON=false
elif [[ "$1" == "--python" ]]; then
    SETUP_NODE=false
fi

cd "$PROJECT_ROOT"

# ============================================
# Node.js Setup
# ============================================
if $SETUP_NODE; then
    print_header "Setting up Node.js environment"

    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js not found. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi

    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [[ "$NODE_VERSION" -lt 18 ]]; then
        print_error "Node.js 18+ required. Found: $(node -v)"
        exit 1
    fi
    print_success "Node.js $(node -v) detected"

    # Install npm dependencies
    echo "Installing npm dependencies..."
    npm install
    print_success "npm dependencies installed"

    # Verify build works
    echo "Verifying build..."
    if npm run build > /dev/null 2>&1; then
        print_success "Build successful"
    else
        print_warning "Build had issues - run 'npm run build' to see details"
    fi
fi

# ============================================
# Python Setup
# ============================================
if $SETUP_PYTHON; then
    print_header "Setting up Python environment"

    # Check Python
    PYTHON_CMD=""
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9 ]]; then
        print_error "Python 3.9+ required. Found: Python $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python $PYTHON_VERSION detected"

    # Create virtual environment
    VENV_DIR="$PROJECT_ROOT/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created at .venv/"
    else
        print_success "Virtual environment already exists"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip > /dev/null
    print_success "pip upgraded"

    # Install the rlbook package in development mode
    echo "Installing rlbook package..."
    pip install -e "$PROJECT_ROOT/code" > /dev/null
    print_success "rlbook package installed"

    # Install test dependencies
    echo "Installing test dependencies..."
    pip install pytest pytest-cov > /dev/null
    print_success "Test dependencies installed"

    # Run tests
    print_header "Running Python tests"
    cd "$PROJECT_ROOT/code"
    if pytest tests/ -v; then
        print_success "All tests passed!"
    else
        print_warning "Some tests failed - check output above"
    fi
    cd "$PROJECT_ROOT"
fi

# ============================================
# Summary
# ============================================
print_header "Setup Complete!"

echo "Next steps:"
echo ""
if $SETUP_NODE; then
    echo "  Frontend development:"
    echo "    npm run dev        # Start dev server at http://localhost:4321"
    echo "    npm run build      # Production build"
    echo ""
fi
if $SETUP_PYTHON; then
    echo "  Python development:"
    echo "    source .venv/bin/activate   # Activate virtual environment"
    echo "    pytest code/tests/          # Run tests"
    echo "    python -m rlbook.examples.train_gridworld  # Run example"
    echo ""
fi
echo "  Documentation:"
echo "    See CLAUDE.md for AI collaboration guide"
echo "    See docs/CONTENT_TYPES.md for content structure"
echo ""
