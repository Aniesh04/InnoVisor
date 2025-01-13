import streamlit as st

# Set the title and description
st.set_page_config(page_title="InnoVisor", page_icon=":bulb:")
st.title("InnoVisor")
st.write("Your AI-powered guide to industry-specific use cases and insights.")

# Layout for horizontally aligned text inputs
col1, col2 = st.columns(2)
with col1:
    company_name = st.text_input("Company Name")
with col2:
    company_url = st.text_input("Company URL")

# Display output when inputs are provided
if company_name and company_url:
    # Placeholder for the use case generation function
    # Here, simulate the function with a dummy output for demonstration
    use_cases = f"Generating tailored use cases for {company_name} based on industry insights..."
    references = "- [Industry Trends Report](https://example.com)\n- [AI Use Case Research](https://example.com)"

    # Display the output in a chat-style message
    st.subheader("Generated Use Cases and References")
    st.write(f"**Company:** {company_name}")
    st.write(f"**Use Cases:**\n{use_cases}")
    st.write(f"**References:**\n{references}")
else:
    st.info("Please enter both the Company Name and Company URL to generate use cases.")
