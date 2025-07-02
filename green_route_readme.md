# 🌿 Green Route Optimization for Supply Chain Logistics

## 🚚 A Streamlit-based Intelligent Tool for Sustainable Delivery Planning

This project helps logistics and supply chain managers **optimize delivery routes** and **minimize carbon emissions** using **real-time APIs**, a **trained machine learning model**, and **interactive geospatial visualization**.

---

## 📌 Features

- 📍 Admin panel to manage delivery locations (Delhi region)
- 🛣 Real-time route generation using OpenRouteService API
- 🌦 Weather integration using OpenWeatherMap API
- 🚦 Traffic data using TomTom API
- ⚖️ Carbon emission prediction using a custom-trained ML model (PyTorch)
- 🗺 Interactive map visualization of optimized delivery route
- 📊 Cargo-specific emissions calculator
- 💡 Actionable green logistics tips
- Built with **Streamlit** and deployed-ready

---

## 📁 Project Structure

```
├── admin.py                  # Admin panel for adding/viewing locations in SQLite
├── app.py                    # Route visualizer with real-time data and map
├── main.py                   # Carbon emission calculator using ML model
├── home.py                   # Streamlit app home and navigation
├── train.py                  # Model training script using synthetic data
├── preprocessing.py          # (Optional) Advanced feature engineering for future model
├── api_clients.py            # API integration for route, weather, traffic
├── requirements.txt          # Python dependencies
├── locations.db              # SQLite database of delivery locations
├── models/
│   ├── carbon_model.pt       # Trained PyTorch model
│   ├── scaler_mean.npy       # Scaler mean values
│   └── scaler_scale.npy      # Scaler scale values
└── utils/
    └── __init__.py           # (Optional) Utility functions folder
```

---

## 🔍 How It Works – Step by Step

### 1. 🚦 Admin Panel (`admin.py`)
- Add new delivery locations (e.g., warehouses, stores) in Delhi
- Stored in local SQLite DB (`locations.db`)

### 2. 🧭 Navigation (`home.py`)
- Sidebar navigation to all app modules:
  - Home
  - Emission Calculator
  - Route Visualization
  - Optimization Tips
  - Admin Panel

### 3. 📊 Carbon Emission Estimation (`main.py`)
- User selects:
  - Start and end locations
  - Vehicle type (Small Van / Medium Truck / Large Truck)
  - Mileage and cargo weight
- App fetches:
  - Distance/duration from OpenRouteService
  - Temperature & humidity from OpenWeatherMap
  - Traffic level from TomTom
- All inputs are fed into a PyTorch ML model to estimate **kg of CO₂**

### 4. 🗺 Route Visualization (`app.py`)
- Similar to `main.py` with additional **folium map**
- Map highlights route, markers, and CO₂ footprint in real-time

### 5. 🧠 Machine Learning Model (`train.py`)
- Synthetic data generator (~200K examples)
- Features:
  - Distance, duration, cargo weight, mileage
  - Traffic level, delay, temperature, humidity
  - One-hot encoded vehicle type
- Model:
  - PyTorch neural network with 3 layers
  - Trained using MSE loss
- Output:
  - `carbon_model.pt` (weights)
  - Scaler files for normalization

### 6. 🌦 Weather, 🚦 Traffic & 🛣 Route APIs (`api_clients.py`)
- `get_route_data`: from OpenRouteService
- `get_weather_data`: from OpenWeatherMap
- `get_traffic_data`: from TomTom

### 7. 🛠 Optional – Advanced Preprocessing (`preprocessing.py`)
- For future version: includes elevation, wind speed, precipitation
- Supports multiple fuel types and advanced feature scaling

---

## 🧠 Sample Model Inputs

- Distance: 25 km  
- Duration: 40 mins  
- Traffic Level: heavy  
- Temperature: 35°C  
- Humidity: 70%  
- Cargo: 3000 kg  
- Mileage: 10 km/l  
- Vehicle Type: Medium Truck

> Estimated Emission: **~17.5 kg CO₂**

---

## 🔐 API Keys (required via Streamlit Secrets)

Configure your Streamlit `secrets.toml` file like:

```toml
# .streamlit/secrets.toml
ORS_API_KEY = "your_openrouteservice_key"
OPENWEATHER_API_KEY = "your_openweather_key"
TOMTOM_API_KEY = "your_tomtom_key"
```

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/Green-Route-Optimization-for-Supply-Chain.git
cd Green-Route-Optimization-for-Supply-Chain

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run home.py
```

---

## 🧪 Train Your Own Model (Optional)

```bash
python train.py
```

This creates:
- `models/carbon_model.pt`
- `models/scaler_mean.npy`
- `models/scaler_scale.npy`

---

## 📌 Dependencies

Listed in `requirements.txt`. Major packages include:

- `streamlit`
- `folium`, `streamlit-folium`
- `openrouteservice`
- `requests`
- `pandas`, `numpy`, `scikit-learn`
- `torch`

---

## 🌐 Demo & Deployment

To deploy this project:

- 📍 [Streamlit Cloud](https://streamlit.io/cloud)
- 🌍 [Hugging Face Spaces](https://huggingface.co/spaces)
- ☁️ AWS / GCP / Azure (for scalable backend + DB)

---

## 🙋‍♂️ Author

👨‍💻 **Ankit Dixit**  
Part of **Edunet ICBP 2.0 Program**  
*"Optimizing logistics for a greener tomorrow"*

---

## 📄 License

MIT License – free to use, distribute, and modify with credit.

---

## 🤝 Contributions

PRs welcome. You can contribute by:

- Adding new city/location support
- Integrating real datasets
- Improving ML accuracy
- Expanding to EVs & alternate fuels

---