import warnings
import streamlit as st

# Suppress Streamlit warnings
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*Session state does not function.*')

import pandas as pd
import numpy as np
import pickle
import random

# --- 1. Load the trained XGBoost model and feature list ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model_and_features():
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except FileNotFoundError:
        st.error("Error: 'xgboost_model.pkl' or 'model_features.pkl' not found. "
                 "Please ensure these files are in the same directory as this script.")
        st.stop()

model, model_features = load_model_and_features()

# --- 2. Function to generate random feature data ---
def generate_random_trip_features(features_list, current_trip_id=None):
    data = {}
    # Realistic ranges for features (aligned with dataset generation logic)
    ranges = {
        'container_age': (5, 25, 'int'),
        'shipment_value': (10000, 25000, 'int'),
        'cumulative_hours': (0.0, 4.0, 'float'),
        'road_type': (1, 3, 'int'),
        'elevation_m': (10.0, 300.0, 'float'),
        'speed_kmh': (0.0, 120.0, 'float'),
        'shock_g': (0.1, 3.5, 'float'),
        'temp_c': (18.0, 42.0, 'float'),
        'humidity_%': (35.0, 85.0, 'float'),
        'oxygen_ppm': (20800.0, 22500.0, 'float'),
        'time_above_28c': (0.0, 4.0, 'float'), # Max duration of trip
        'vibration_event_count': (0, 10, 'int'),
        'shock_anomaly': (0, 1, 'int'),
        'temperature_anomaly': (0, 1, 'int'),
        'O2_anomaly': (0, 1, 'int')
    }

    if current_trip_id is None:
        # Generate a unique trip ID for the initial trip display
        current_trip_id = f"trip_{np.random.randint(1000, 9999)}"
    
    # Store trip_id separately since it's not a feature for the model
    trip_id = current_trip_id

    # Generate features for the model
    for feature in features_list:
        min_val, max_val, f_type = ranges.get(feature, (0, 1, 'float'))
        if f_type == 'int':
            data[feature] = [np.random.randint(min_val, max_val + 1)]
        elif f_type == 'float':
            data[feature] = [np.random.uniform(min_val, max_val)]

    df_features = pd.DataFrame(data)
    
    # Ensure the order of columns matches the model_features
    df_features = df_features[features_list]
    
    return df_features, trip_id

# --- 3. Function to generate automatic alternative routes ---
def generate_alternative_routes(base_features, num_routes=5):
    """Generate realistic alternative routes by perturbing features"""
    
    routes_data = []
    routes_info = []
    
    # Define realistic perturbations for each route type
    route_templates = [
        {
            'name': 'Cooler Route',
            'description': 'Lower temperature, reduced heat exposure',
            'perturbations': {
                'temp_c': (-4, -8),
                'time_above_28c': (-1.5, -0.5),
                'temperature_anomaly': -1,
                'speed_kmh': (-10, -5),
                'cumulative_hours': (0.1, 0.3)  # Slightly longer but cooler
            }
        },
        {
            'name': 'Smooth Road Route',
            'description': 'Better road conditions, less vibration',
            'perturbations': {
                'shock_g': (-0.8, -0.3),
                'vibration_event_count': (-3, -1),
                'shock_anomaly': -1,
                'road_type': -1,  # Better road type (lower number)
                'speed_kmh': (5, 15)  # Can go faster on smooth roads
            }
        },
        {
            'name': 'High Elevation Route',
            'description': 'Higher altitude, cooler temperatures',
            'perturbations': {
                'elevation_m': (80, 150),
                'temp_c': (-3, -6),
                'oxygen_ppm': (-200, -100),
                'speed_kmh': (-15, -8),
                'cumulative_hours': (0.2, 0.4)
            }
        },
        {
            'name': 'Quick Route',
            'description': 'Faster delivery but potentially rougher',
            'perturbations': {
                'speed_kmh': (15, 25),
                'cumulative_hours': (-0.5, -0.2),
                'shock_g': (0.1, 0.4),
                'vibration_event_count': (1, 3),
                'temp_c': (1, 3)  # Slightly warmer due to faster travel
            }
        },
        {
            'name': 'Balanced Route',
            'description': 'Optimal balance of all factors',
            'perturbations': {
                'temp_c': (-2, -4),
                'shock_g': (-0.3, -0.1),
                'speed_kmh': (-5, 5),
                'oxygen_ppm': (-50, -20),
                'vibration_event_count': (-1, 0),
                'time_above_28c': (-0.5, -0.2)
            }
        }
    ]
    
    # Define ranges for feature clipping
    ranges = {
        'container_age': (5, 25, 'int'),
        'shipment_value': (10000, 25000, 'int'),
        'cumulative_hours': (0.0, 4.0, 'float'),
        'road_type': (1, 3, 'int'),
        'elevation_m': (10.0, 300.0, 'float'),
        'speed_kmh': (0.0, 120.0, 'float'),
        'shock_g': (0.1, 3.5, 'float'),
        'temp_c': (18.0, 42.0, 'float'),
        'humidity_%': (35.0, 85.0, 'float'),
        'oxygen_ppm': (20800.0, 22500.0, 'float'),
        'time_above_28c': (0.0, 4.0, 'float'),
        'vibration_event_count': (0, 10, 'int'),
        'shock_anomaly': (0, 1, 'int'),
        'temperature_anomaly': (0, 1, 'int'),
        'O2_anomaly': (0, 1, 'int')
    }
    
    base_dict = base_features.iloc[0].to_dict()
    
    for i, template in enumerate(route_templates[:num_routes]):
        # Create perturbed features
        perturbed_dict = base_dict.copy()
        
        for feature, change in template['perturbations'].items():
            current_value = perturbed_dict[feature]
            
            if feature.endswith('_anomaly'):
                if change == -1:
                    perturbed_dict[feature] = 0
                elif change == 1:
                    perturbed_dict[feature] = 1
                else:
                    perturbed_dict[feature] = np.random.randint(0, 2)
            elif isinstance(change, tuple):
                # Random change within range
                min_change, max_change = change
                perturbed_dict[feature] = current_value + np.random.uniform(min_change, max_change)
            else:
                # Direct change
                perturbed_dict[feature] = current_value + change
            
            # Clip to valid ranges
            if feature in ranges:
                min_val, max_val, f_type = ranges[feature]
                perturbed_dict[feature] = np.clip(perturbed_dict[feature], min_val, max_val)
                if f_type == 'int':
                    perturbed_dict[feature] = int(perturbed_dict[feature])
        
        # Prepare for prediction
        feature_values = [perturbed_dict[feature] for feature in model_features]
        pred_df = pd.DataFrame([feature_values], columns=model_features)
        predicted_loss = model.predict(pred_df)[0]
        
        # Store route data
        route_dict = perturbed_dict.copy()
        route_dict['Route'] = f"Route {i+1}: {template['name']}"
        route_dict['Description'] = template['description']
        route_dict['Predicted_Loss'] = predicted_loss
        
        routes_data.append(route_dict)
        
        # Store route info for summary
        routes_info.append({
            'Route': f"Route {i+1}",
            'Name': template['name'],
            'Description': template['description'],
            'Predicted_Loss': predicted_loss
        })
    
    return routes_data, pd.DataFrame(routes_info)

# --- 4. GUI Layout ---
st.set_page_config(layout="wide", page_title="Olive Oil Transport Rerouting Simulator")

st.title("🚚 Olive Oil Transport Rerouting Simulator")
st.markdown("This application simulates rerouting decisions for olive oil transport based on an XGBoost model's predicted estimated loss.")

st.sidebar.header("Simulation Controls")
if st.sidebar.button("Generate New Simulation"):
    st.session_state.initial_trip_df = None
    st.session_state.initial_trip_id = None
    st.session_state.alternative_routes = None
    st.rerun()

# Sidebar configuration
st.sidebar.subheader("Configuration")
num_routes = st.sidebar.slider("Number of Alternative Routes", 3, 10, 5)

# Generate or retrieve initial trip data
if 'initial_trip_df' not in st.session_state or st.session_state.initial_trip_df is None:
    initial_trip_df, initial_trip_id = generate_random_trip_features(model_features)
    st.session_state.initial_trip_df = initial_trip_df
    st.session_state.initial_trip_id = initial_trip_id
    # Generate alternative routes
    alternative_routes, routes_summary = generate_alternative_routes(initial_trip_df, num_routes)
    st.session_state.alternative_routes = alternative_routes
    st.session_state.routes_summary = routes_summary
else:
    initial_trip_df = st.session_state.initial_trip_df.copy()
    initial_trip_id = st.session_state.initial_trip_id
    alternative_routes = st.session_state.alternative_routes
    routes_summary = st.session_state.routes_summary

# --- 5. Display Original Trip ---
st.header("📋 Original Trip Scenario")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Trip ID", initial_trip_id)
with col2:
    st.write("**Initial Conditions:**")

st.dataframe(initial_trip_df)

# Predict loss for the initial trip
initial_predicted_loss = model.predict(initial_trip_df)[0]
st.success(f"### Predicted Estimated Loss for Original Trip: **€{initial_predicted_loss:.2f}**")

st.markdown("---")

# --- 6. Display All Alternative Routes ---
st.header("🔄 Rerouting Alternatives")
st.write(f"Exploring {num_routes} automatically generated alternative routes:")

# Display routes summary in an expandable section
with st.expander("📊 Routes Overview (Click to expand)", expanded=True):
    # Sort by predicted loss (ascending - lower is better)
    sorted_summary = routes_summary.sort_values('Predicted_Loss').reset_index(drop=True)
    
    # Create columns for display
    cols = st.columns(3)
    for idx, row in sorted_summary.iterrows():
        col_idx = idx % 3
        with cols[col_idx]:
            # Color code based on performance
            if idx == 0:  # Best route
                st.markdown(f"**🥇 {row['Route']}**")
                st.metric("Predicted Loss", f"€{row['Predicted_Loss']:.2f}", 
                         f"Best", delta_color="inverse")
            elif idx == 1:  # Second best
                st.markdown(f"**🥈 {row['Route']}**")
                improvement = initial_predicted_loss - row['Predicted_Loss']
                if improvement > 0:
                    st.metric("Predicted Loss", f"€{row['Predicted_Loss']:.2f}", 
                             f"Save €{improvement:.2f}", delta_color="inverse")
                else:
                    st.metric("Predicted Loss", f"€{row['Predicted_Loss']:.2f}")
            else:
                st.markdown(f"**{row['Route']}**")
                improvement = initial_predicted_loss - row['Predicted_Loss']
                if improvement > 0:
                    st.metric("Predicted Loss", f"€{row['Predicted_Loss']:.2f}", 
                             f"Save €{improvement:.2f}", delta_color="inverse")
                else:
                    st.metric("Predicted Loss", f"€{row['Predicted_Loss']:.2f}")
            
            st.caption(f"*{row['Description']}*")

# Detailed view of all routes
st.subheader("📈 Detailed Route Comparison")

# Create tabs for each route
tabs = st.tabs([f"Route {i+1}" for i in range(num_routes)])

for idx, tab in enumerate(tabs):
    with tab:
        route_data = alternative_routes[idx]
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### {route_data['Route'].split(': ')[1]}")
            st.write(f"**{route_data['Description']}**")
            
            # Show key metrics
            improvement = initial_predicted_loss - route_data['Predicted_Loss']
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Predicted Loss", f"€{route_data['Predicted_Loss']:.2f}")
            with metrics_col2:
                if improvement > 0:
                    st.metric("Savings vs Original", f"€{improvement:.2f}", 
                             f"{(improvement/initial_predicted_loss*100):.1f}%")
                else:
                    st.metric("Additional Cost", f"€{-improvement:.2f}", 
                             f"{(-improvement/initial_predicted_loss*100):.1f}%")
        
        with col2:
            # Show feature changes
            st.markdown("#### Feature Changes vs Original")
            
            # Get original values
            original_dict = initial_trip_df.iloc[0].to_dict()
            
            # Create comparison
            comparison_data = []
            for feature in model_features:
                original_val = original_dict[feature]
                new_val = route_data[feature]
                change = new_val - original_val
                percent_change = (change / original_val * 100) if original_val != 0 else 0
                
                comparison_data.append({
                    'Feature': feature,
                    'Original': f"{original_val:.2f}" if isinstance(original_val, float) else original_val,
                    'New': f"{new_val:.2f}" if isinstance(new_val, float) else new_val,
                    'Change': f"{change:+.2f}" if isinstance(change, float) else f"{change:+d}",
                    '% Change': f"{percent_change:+.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Color code improvements (green) and deteriorations (red)
            def color_change(val):
                try:
                    change = float(val.replace('%', '').replace('+', ''))
                    if change < 0:
                        return 'color: green'  # Negative change is good for loss reduction
                    elif change > 0:
                        return 'color: red'
                    else:
                        return ''
                except:
                    return ''
            
            styled_df = comparison_df.style.applymap(color_change, subset=['Change', '% Change'])
            st.dataframe(styled_df, use_container_width=True)

# --- 7. Optimal Route Recommendation ---
st.markdown("---")
st.header("🏆 Optimal Route Recommendation")

# Find optimal route
if routes_summary is not None and not routes_summary.empty:
    optimal_route_idx = routes_summary['Predicted_Loss'].idxmin()
    optimal_route = routes_summary.iloc[optimal_route_idx]
    optimal_route_details = alternative_routes[optimal_route_idx]
    
    # Display optimal route
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("**Recommended Route**", optimal_route['Route'])
    
    with col2:
        improvement = initial_predicted_loss - optimal_route['Predicted_Loss']
        if improvement > 0:
            st.success(f"### Save €{improvement:.2f} ({improvement/initial_predicted_loss*100:.1f}%)")
        else:
            st.warning(f"### Additional cost: €{-improvement:.2f}")
    
    with col3:
        st.metric("Predicted Loss", f"€{optimal_route['Predicted_Loss']:.2f}")
    
    st.info(f"**Why this route?** {optimal_route['Description']}")
    
    # Show key improvements
    st.subheader("Key Improvements in Optimal Route:")
    
    # Get top 3 improved features
    original_dict = initial_trip_df.iloc[0].to_dict()
    improvements = []
    
    for feature in model_features:
        if feature in ['shock_anomaly', 'temperature_anomaly', 'O2_anomaly']:
            # For anomaly features, reduction is good
            if optimal_route_details[feature] < original_dict[feature]:
                improvements.append((feature, "Anomaly eliminated", "🟢"))
        elif feature in ['temp_c', 'shock_g', 'vibration_event_count']:
            # For these, reduction is good
            change = optimal_route_details[feature] - original_dict[feature]
            if change < 0:
                percent = abs(change / original_dict[feature] * 100)
                improvements.append((feature, f"Reduced by {abs(change):.1f} ({percent:.0f}%)", "📉"))
    
    # Display improvements
    if improvements:
        cols = st.columns(min(3, len(improvements)))
        for idx, (feature, desc, icon) in enumerate(improvements[:3]):
            with cols[idx % 3]:
                st.markdown(f"{icon} **{feature.replace('_', ' ').title()}**")
                st.caption(desc)

# --- 8. Export Results ---
st.markdown("---")
st.subheader("📥 Export Results")

if st.button("📋 Export All Route Data to CSV"):
    # Combine all data
    all_data = []
    
    # Add original trip
    original_row = initial_trip_df.iloc[0].to_dict()
    original_row['Route'] = 'Original'
    original_row['Description'] = 'Initial trip scenario'
    original_row['Predicted_Loss'] = initial_predicted_loss
    all_data.append(original_row)
    
    # Add alternative routes
    all_data.extend(alternative_routes)
    
    # Create DataFrame and export
    export_df = pd.DataFrame(all_data)
    
    # Reorder columns
    cols = ['Route', 'Description', 'Predicted_Loss'] + model_features
    export_df = export_df[cols]
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"rerouting_simulation_{initial_trip_id}.csv",
        mime="text/csv"
    )

# Add some styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stSuccess {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)