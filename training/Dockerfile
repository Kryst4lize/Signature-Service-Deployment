FROM 10.254.144.152/tessel/training-service-base:0.0.1

COPY pip.conf /etc/pip.conf

# 2. INJECT AND INSTALL TENSORRT OFFLINE
WORKDIR /tmp/wheels
COPY ../libs_local/*.whl /tmp/wheels/
RUN pip install /tmp/wheels/*.whl && rm -rf /tmp/wheels


# 4. INSTALL EVERYTHING ELSE
WORKDIR /app
COPY requirements.txt .

# Install the remaining requirements (Make sure tensorrt is NOT in this file!)
RUN pip install -r requirements.txt