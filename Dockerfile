# 1. Start with a lightweight Linux + Python 3.10 base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (for better caching)
COPY requirements.txt .

# 4. Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project code
COPY . .

# 6. Command to run when the container starts
CMD ["python", "src/scripts/test_setup.py"]
