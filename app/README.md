# 🛰️ OceanGuard AI: Interactive Mapping Platform

<p align="center">
  <img src="https://img.shields.io/badge/App-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Framework">
  <img src="https://img.shields.io/badge/Model-Gemma%203n-blue?style=for-the-badge&logo=google-gemini" alt="Model">
  <img src="https://img.shields.io/badge/Optimized%20by-Unsloth-green?style=for-the-badge" alt="Unsloth">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

---

### 💡 The Solution: Actionable Ocean Intelligence

**OceanGuard AI** is an interactive platform that uses Google's multimodal **Gemma 3n** model to turn underwater imagery into real-time conservation insights. This application allows researchers, environmental agencies, and citizen scientists to detect, map, and analyze marine debris pollution instantly.

---

## 🎥 Live Demo

<p align="center">
  <em>(Add a GIF of the application in action here)</em>
  <br>
  <img src="placeholder.gif" alt="OceanGuard AI Demo GIF" width="80%">
</p>

---

## ✨ Application Features

This interactive demo showcases the full power of the OceanGuard AI pipeline:

-   **📸 Image Upload & Geolocation**: Upload multiple underwater images. The app automatically extracts GPS data from EXIF tags or assigns a location from a global catalog of dive sites.
-   **🤖 One-Click AI Analysis**: With a single button press, Gemma analyzes all uploaded images to detect debris, identify materials, and generate precise bounding boxes.
-   **🗺️ Interactive 3D Map**: Visualize pollution hotspots on a dynamic 3D map. Each point is color-coded by its **Ecosystem Health Score** and sized by debris density.
-   **📊 Dynamic Dashboards**: View aggregate statistics for your entire project or filter down to a specific location to analyze debris types and concentrations.
-   **📄 Detailed Individual Reports**: Dive into any single image to see the original photo, the AI-annotated version, and a detailed breakdown of each piece of debris found.
-   **✍️ AI-Generated Executive Summaries**: Generate multilingual (EN, ES, FR, etc.) summary reports for any location, complete with an environmental impact assessment and actionable recommendations.
-   **📥 Data Export**: Download your complete analysis, including all detections and coordinates, as a CSV file for further research.

---

## 🏁 How to Use the Demo

1.  **Upload Images**: Use the sidebar to upload one or more `.jpg`, `.jpeg`, or `.png` underwater images.
2.  **Analyze**: Click the **"Analyze Project Images"** button. A progress bar will show the analysis status.
3.  **Explore the Dashboard**:
    -   Interact with the **3D map** to see the geographic distribution of pollution.
    -   Use the **location dropdown** to view specific statistics.
4.  **View Individual Reports**:
    -   Navigate to the **"Individual Reports"** tab.
    -   Select an image from the dropdown to see a side-by-side comparison and an impact panel.
5.  **Generate AI Reports**:
    -   In the sidebar, choose a language for your report.
    -   Click **"Generate Reports by Location"**.
    -   View the generated reports in the **"AI Generated Reports"** tab.

---

## 💻 Local Installation & Setup

To run this Streamlit application on your local machine:

1.  **Clone the Repository**:
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Set up a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**: A `requirements.txt` file should be created from the project's dependencies.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**:
    ```bash
    streamlit run app/main.py
    ```

---

## 🤝 Contact

-   **Project Lead**: [Alejandro Sánchez Ferrer on LinkedIn](https://www.linkedin.com/in/alejandro-sanchez-ferrer/)
-   **Team**: © 2025 Saflex Team