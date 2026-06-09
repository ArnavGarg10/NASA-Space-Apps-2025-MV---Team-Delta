# 🦈 Project Delta: Shark Foraging Prediction & Real-Time Tracking Tag
### **Winner: 1st Place (NASA Space Apps 2025, Mountain View)** ### ** and Nominated For Global Judging **

---

## 📖 Project Overview
Project Delta addresses the **NASA Space Apps Challenge 2025: Sharks from Space**. 

Apex marine predators like sharks are critical to maintaining ocean ecosystems, yet gathering real-time data on their exact foraging habits and pathways remains extremely difficult. Our project solves this by introducing a dual-system framework:
1. **A Predictive Mathematical Model** built using real NASA satellite data (**SWAT** and **PACE OCI** missions) to locate warm ocean eddies and chlorophyll/phytoplankton hotspots, calculating exactly where a shark will travel to forage using Haversine formulas.
2. **A Hardware Tracking Tag Embedded with Edge AI** designed entirely from scratch using a Raspberry Pi, an accelerometer, and a camera to physically capture when and what a shark is eating, then streaming that data back to simulated ocean buoys.

---

## 🛠 Technical Architecture

### 1. Mathematical Framework & Hotspot Prediction
Sharks utilize warm ocean eddies as navigation conduits because these eddies experience large phytoplankton blooms, attracting smaller prey (fish). 
* **Data Ingestion**: The system processes NetCDF files directly from two active NASA Earth observation missions:
  * **SWAT (Surface Water and Ocean Topography)**: Maps ocean surface height anomalies to identify warm ocean eddies.
  * **PACE OCI (Ocean Color Instrument)**: Tracks chlorophyll-a and phytoplankton concentration levels globally.
* **Haversine Distance Mapping**: Given an incoming GPS coordinate of a shark, the backend executes multiple Haversine distance calculations across every extracted eddy coordinate to find the closest active marine system:
  $$\text{d} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
* **Visualization Pipeline**: Built using `matplotlib` and `cartopy` in Python, the pipeline generates high-fidelity spatial plots showing the shark's current location (represented by a black star), the predicted foraging eddy destination (red dot), and scalar fields representing chlorophyll density.

### 2. Custom Edge Hardware Tag
A fully functional hardware prototype built to detect and record feeding events on the edge.
* **The Compute Engine**: A **Raspberry Pi 4** acts as the central processor executing local scripts.
* **Foraging Detection**: An **accelerometer** is mapped to track rapid mechanical movements of the shark's jaw. When the jaw opens to feed, a threshold trigger initializes data capture.
* **Visual Capture**: A **hardware camera module** triggers at the precise moment of jaw acceleration, taking snapshots of the prey in real time.
* **Data Logging**: Every captured event creates a timestamped record saved directly to a `photo_locations.csv` file mapping the file name, precise time, and current coordinate telemetry.

### 3. Edge AI & Data Relay
* **The Relay Mechanism**: In our conceptual model, the Raspberry Pi tag uses **SCP (Secure Copy Protocol)** to securely transmit images and CSV data logs over an IP network to a local Linux PC simulating an ocean data-relay buoy. 
* **Edge Image Classification**: Once received by the buoy interface, an image detection model evaluates the snapshots to classify what the shark consumed (e.g., identifying fish), sorting and saving the data cleanly for marine biologists.

---

## 🖥 User Interface & Features
* **Predictive Coordinates Interface**: Researchers input individual coordinates to immediately calculate the predicted destination, distance to target (in km), and projected travel time.
* **Dynamic Plot Generation**: Provides downloadable maps rendering historical data overlays with clear markers for localized analysis.
* **Batch CSV Upload Engine**: Supports high-throughput pipelines where researchers can upload a `shark_locations.csv` file of multi-shark tracking histories and download a processed batch results dataset instantly.

---

## 🚀 The Technical Stack
* **Frontend/Interface**: Flask, HTML5, Custom CSS, JavaScript
* **Data Processing & Analytics**: Python, NumPy, Pandas, NetCDF4
* **Mapping & Graphics**: Matplotlib, Cartopy
* **Edge AI & Computer Vision**: OpenCV, Image Classification Frameworks
* **Hardware Layer**: Raspberry Pi 4, Digital Accelerometers, Pi Camera Module
* **Data Sources**: NASA Earth Observations (SWAT Surface Height & PACE OCI Aquamotus Chlorophyll data)

---

## 💡 Future Roadmap & Real-World Scaling
While our hackathon prototype proved the core mechanics, we designed this project with future iterations in mind:
* **Acoustic & Satellite Telemetry**: Replacing the conceptual network-layer SCP data transfer with acoustic pings and radio/satellite arrays viable for deep-sea transmission.
* **Dedicated GPS Hardware**: Integrating low-power hardware GPS modules directly onto the tag to bypass IP-based simulation for true field telemetry.
* **Model Optimization**: Utilizing the real-world captured feeding data to reinforce and retrain our predictive mathematical models, creating a continuous feedback loop between predicted tracks and actual physical catches.

---

## 👥 Team Delta
* **Nikhil Konduru** — AI Engineer & Hardware Architect
* **Arnav Garg** — Frontend Developer & ML Engineer
