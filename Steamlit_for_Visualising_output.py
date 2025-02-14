import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io  

# Load evaluation results
with open("best_models_summary.pkl", "rb") as file:
    best_models_summary = pickle.load(file)

def plot_actual_vs_predicted(actual, predicted, title, max_points=100):
    """Plot the actual vs predicted values and return the figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Limit the number of points plotted to avoid clutter
    if len(actual) > max_points:
        step = len(actual) // max_points
        actual = actual[::step]
        predicted = predicted[::step]
    
    ax.plot(actual, label='Actual', color='b', linewidth=1.5)
    ax.plot(predicted, label='Predicted', color='r', linestyle='--', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Price')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()  
    st.pyplot(fig)
    
    return fig  


def main():
    st.title("Financial Market Model Evaluation")
    
    # Select market
    market = st.selectbox("Select Market", ["Stocks", "Forex", "Cryptocurrencies"])
    
    # Select symbol
    symbols = list(best_models_summary[market].keys())
    symbol = st.selectbox("Select Symbol", symbols)
    
    # Display model metrics
    st.header(f"Best Models for {symbol}")
    metrics_df = best_models_summary[market][symbol]
    st.write(metrics_df)

    # Select model to view actual vs predicted plot
    model = st.selectbox("Select Model for Plotting", metrics_df.index)
    
    # Plot actual vs predicted graph
    actual = metrics_df.loc[model, "Actual"]
    predicted = metrics_df.loc[model, "Predictions"]
    
    # Generate the plot
    fig = plot_actual_vs_predicted(actual, predicted, f"Actual vs Predicted for {symbol} using {model}")
    
    # Add download option
    img_buffer = io.BytesIO()  
    fig.savefig(img_buffer, format='png')  
    img_buffer.seek(0)  
    
    st.download_button(
        label="Download plot as PNG",
        data=img_buffer,
        file_name=f"{symbol}_{model}_plot.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()
