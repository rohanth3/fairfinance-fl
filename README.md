# FairFinance: Enterprise Federated Fraud Detection Framework

**Developed and Optimized by: Rohanth Ganesha C**

## Project Summary
FairFinance is a production-grade, privacy-preserving federated learning (FL) system specifically engineered for the financial services sector. The framework allows decentralized banking institutions to collectively train an advanced fraud detection model on their local datasets without ever transferring raw transaction data across the network. By leveraging differential privacy, secure model management, and explainable AI, FairFinance provides a secure alternative to risky centralized data pooling.

## Architecture
FairFinance follows a highly concurrent **Client-Server Federated Learning Architecture**:
1. **Central Aggregation Server (`server.py`)**: Acts as the master orchestrator. It holds the global model but possesses NO access to any training data. It accepts encrypted weight updates from connected clients, securely aggregates them (FedAvg) using background threading to avoid bottlenecking, and distributes the improved global model back to all nodes.
2. **Federated Clients (`client.py`)**: Represent individual banks or financial institutions. Each client holds a local, private dataset of transactions. Clients download the global model, train it locally on their data utilizing automatic GPU acceleration (`cuda`/`mps`), and apply strict Differential Privacy algorithms before returning their data.
3. **Monitoring Dashboard (`app.py`)**: A real-time, responsive Streamlit interface that connects directly to the server's tracking history to visualize global model accuracy, fairness metrics, client contributions, and overall system health.

## Workflow
1. **Initialization**: The Server is started, locking in configuration parameters and initializing a generic global neural network.
2. **Client Connection**: Banks (Clients) connect to the server via secure TCP sockets. 
3. **Local Training**: Clients download the global model and train it on their local transaction data for a predefined number of epochs.
4. **Privacy Layer**: Before leaving the client, the updated model weights are subjected to **Local Differential Privacy (LDP)**. Gradients are clipped and Laplacian/Gaussian noise is injected natively on the GPU to prevent adversarial reverse-engineering.
5. **Secure Transmission**: The noisy weights are serialized using `pickle` and dispatched to the server.
6. **Aggregation**: The Server receives client updates concurrently, aggregates them using a Byzantine-robust algorithm (Accuracy-Weighted Aggregation), and applies the changes to the global model.
7. **Broadcast**: The new, smarter global model is sent back to all clients to repeat the loop.

## Tech Stack Used
* **Deep Learning Framework**: PyTorch (NN layers, GPU operations, native Tensor DP noise injection)
* **Frontend UI & Visualization**: Streamlit, Plotly (Dynamic charts, modern CSS gradient styling)
* **Networking Protocol**: Python `socket` library (TCP), `pickle` for model serialization
* **Data Processing**: Pandas, NumPy
* **Concurrency**: `threading` (Concurrent client uploads and non-blocking dashboard UI rendering via `@st.fragment`)

## Output Screenshots

Here are the results and interfaces from the FairFinance system in action:

<details>
<summary><strong>Click to view all Dashboard & Terminal Output Screenshots</strong></summary>

### System Dashboard Overview
![Dashboard 1](Output%20Screenshots/SS1.png)

### Client Contribution & Statistics
![Dashboard 2](Output%20Screenshots/SS2.png)

### Training Progress & Accuracy
![Dashboard 3](Output%20Screenshots/SS3.png)

### System Performance & Fairness
![Dashboard 4](Output%20Screenshots/SS4.png)

### Network Contributions
![Dashboard 5](Output%20Screenshots/SS5.png)

### Extended Analytics
![Dashboard 6](Output%20Screenshots/SS6.png)
![Dashboard 7](Output%20Screenshots/SS7.png)
![Dashboard 8](Output%20Screenshots/SS8.png)
![Dashboard 9](Output%20Screenshots/SS9.png)
![Dashboard 10](Output%20Screenshots/SS10.png)
![Dashboard 11](Output%20Screenshots/SS11.png)
![Dashboard 12](Output%20Screenshots/SS12.png)

### Terminal Output
![Server](Output%20Screenshots/SS13.png)
![Clint](Output%20Screenshots/SS14.png)

</details>


## Data Privacy and Compliance Statement
This software is designed to facilitate compliance with data privacy laws by ensuring that no Personally Identifiable Information (PII) is ever transmitted. The noise addition meets modern differential privacy standards for ε-differential privacy, suitable for production banking environments.
