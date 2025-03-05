import streamlit as st
# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="Women's Financial Empowerment App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)
import pandas as pd
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables if using .env file
load_dotenv()

# Import your existing modules here
from iAccelerate2 import DataManager, FinancialAnalyzer, Module3App, Module4App, Module5Investment, Module6LearnTest

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px !important;
        color: #4EA8DE;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .success-message {
        background-color: #D4EDDA;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .error-message {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .info-card {
        background-color: dark-blue;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #4EA8DE;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton button:hover {
        background-color: #3B82F6;
    }
    .auth-buttons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Main FinancialApp Class ---
class FinancialApp:
    def __init__(self, credentials_file: str, spreadsheet_id: str, cohere_api_key: str, investment_spreadsheet_id: str):
        self.data_manager = DataManager(credentials_file, spreadsheet_id)
        self.analyzer = FinancialAnalyzer(cohere_api_key)
        self.module3 = Module3App(cohere_api_key)
        self.module4 = Module4App()
        self.module5 = Module5Investment(credentials_file, investment_spreadsheet_id, cohere_api_key)
        self.module6 = Module6LearnTest()
        self.cohere_api_key = cohere_api_key  # Store the API key

    def register_user(self) -> None:
        st.markdown('<p class="section-title">📝 Registration</p>', unsafe_allow_html=True)

        user_data = self.data_manager.get_user_details_streamlit()
        if user_data:
            try:
                success = self.data_manager.save_user_data(user_data)  # Save to Google Sheets
                if success:
                    st.markdown('<div class="success-message">✨ Registration successful! You can now login.</div>', unsafe_allow_html=True)
                    st.session_state.registration_success = True
                    st.session_state.auth_action = "login"
                    st.rerun()
                else:
                    st.error("Registration failed while saving data.")

            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
                st.session_state.registration_success = False


    def authenticate_user(self) -> str:
        st.markdown('<p class="section-title">🔐 Login</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login 🚀")
        
        if submit_button:
            if not email or not password:
                st.error("Please fill in all required fields.")
                return None
            
            try:
                existing_users = self.data_manager.sheet.get_all_records()
                for user in existing_users:
                    if user['email'] == email and user['password'] == password:
                        st.session_state.logged_in = True
                        st.session_state.username = user['username']
                        st.markdown(f'<div class="success-message">Login successful! Welcome back, {user["username"]} 👋</div>', unsafe_allow_html=True)
                        st.balloons()
                        # Set the default module to financial insights after login
                        st.session_state.current_module = "financial_insights"
                        st.rerun()
                        return user['username']
                
                st.markdown('<div class="error-message">❌ Invalid email or password.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Login error: {str(e)}")
            
            return None

    def _show_financial_insights(self, username: str) -> None:
        st.markdown('<p class="section-title">📊 Financial Insights</p>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing your financial data..."):
            user_data = self.data_manager.get_user_financial_data(username)
            if isinstance(user_data, pd.DataFrame) and not user_data.empty:
                user_dict = user_data.iloc[0].to_dict()
                user_dict['categories'] = {
                    'Food': float(user_dict['Food']),
                    'Rent': float(user_dict['Rent']),
                    'Utilities': float(user_dict['Utilities']),
                    'Entertainment': float(user_dict['Entertainment']),
                    'Loan Payments': float(user_dict['Loan Payments']),
                    'Savings': float(user_dict['Savings']),
                    'Others': float(user_dict['Others'])
                }
                
                # Expenses pie chart
                st.markdown("### 💸 Expense Breakdown")
                expense_data = user_dict['categories']
                expense_df = pd.DataFrame({
                    'Category': list(expense_data.keys()),
                    'Amount': list(expense_data.values())
                })
                st.bar_chart(expense_df.set_index('Category'))
                
                # Generate insights
                data_key = f"{username}_financial_insights"
                report, visualizations, user_data = self.analyzer.get_financial_insights(data_key, json.dumps(user_dict))
                
                st.markdown("### 🧠 AI-Powered Insights")
                st.markdown(f'<div class="info-card">{report}</div>', unsafe_allow_html=True)
                
                # Income vs Expenses
                st.markdown("### 💰 Income vs Expenses")
                income = float(user_dict['monthly_income'])
                total_expenses = sum(expense_data.values()) - float(user_dict['Savings'])
                remaining = income - total_expenses
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Monthly Income", f"₹{income:,.2f}", "100%")
                col2.metric("Total Expenses", f"₹{total_expenses:,.2f}", f"{total_expenses/income:.1%}")
                col3.metric("Remaining", f"₹{remaining:,.2f}", f"{remaining/income:.1%}")

            else:
                st.error("No financial data found. Please update your profile.")

    def run_module3(self, username):

        df = self.data_manager.get_user_financial_data(username)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Call the run_module3 method of the Module3App instance
            self.module3.run_module3(df, self.cohere_api_key)
        else:
            st.error("No financial data found for financial planning. Please update your profile.")

    def run_module4(self):

        # Adapt Module4App to work with Streamlit
        self.module4.run_module4()

    def run_module5(self, username):

        # Adapt Module5Investment to work with Streamlit
        self.module5.run_module5()

    def run_module6(self):  # Corrected name to match the initialization

        # Adapt Module6LearnTest to work with Streamlit
        # Create an instance here
        module6_instance = Module6LearnTest()
        module6_instance.run_module6_streamlit()

# Initialize the application once
#@st.cache_resource  # REMOVED CACHING DECORATOR
def init_app():
    return FinancialApp(
        credentials_file="iAccelerate-credentials.json",
        spreadsheet_id="1Q0qLnSe9WIUB2ZjwbeonbGJK80Il37eNJr7B2lA2Lkc",
        cohere_api_key="j7uTsOOCsQS99XqLFRUFWHWzWQzADufa6AuHWxXU",
        investment_spreadsheet_id="1LAI3db38-v8YBl1P-F5EI57AiMms5DYMz80NUJN0SF0"
    )

app = init_app()

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_module' not in st.session_state:
    st.session_state.current_module = None
if 'auth_action' not in st.session_state:
    st.session_state.auth_action = None

# Main app logic
def main():
    # Application title - always show this regardless of login state
    st.markdown('<h1 class="main-title">👩‍💼 Women\'s Financial Empowerment App 💪</h1>', unsafe_allow_html=True)
    
    # Handle logged-in state
    if st.session_state.logged_in:
        # Show welcome message in the main area
        st.markdown(f'<div class="info-card">Welcome, {st.session_state.username}! Use the sidebar to navigate through different features.</div>', unsafe_allow_html=True)
        
        # Initialize sidebar for logged-in users
        with st.sidebar:
            st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")
            st.sidebar.markdown("---")
            
            module_options = {
                "financial_insights": "📊 Financial Insights",
                "financial_planning": "🔮 Financial Planning",
                "calculators": "🧮 SIP & EMI Calculators",
                "investment": "📈 Investment Portfolio",
                "learn": "📚 Learn & Test",
                "logout": "🚪 Logout"
            }
        
        selected_module = st.sidebar.radio("📌 Select a Module", list(module_options.keys()), format_func=lambda x: module_options[x])
        st.session_state.current_module = selected_module
        if selected_module == "financial_insights":
            app._show_financial_insights(st.session_state.username)
        elif selected_module == "financial_planning":
            app.run_module3(st.session_state.username)
        elif selected_module == "calculators":
            app.run_module4()
        elif selected_module == "investment":
            app.run_module5(st.session_state.username)
        elif selected_module == "learn":
            app.run_module6()
        elif selected_module == "logout":
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.current_module = None
            st.session_state.auth_action = None
            st.rerun()

    
    else:
        # Welcome message for non-logged in users
        st.markdown("""
        <div class="info-card">
            <h3>🌟 Welcome to Women's Financial Empowerment App! 🌟</h3>
            <p>This app helps women take control of their finances through personalized insights, 
            planning tools, and financial education.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display authentication options if no action is selected yet
        if st.session_state.auth_action is None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Register", use_container_width=True):
                    st.session_state.auth_action = "register"
                    st.rerun()
            
            with col2:
                if st.button("🔐 Login", use_container_width=True):
                    st.session_state.auth_action = "login"
                    st.rerun()
        
        # Handle authentication action based on button clicks
        if st.session_state.auth_action == "register":
            app.register_user()
            # Back button
            if st.button("← Back to Main Menu"):
                st.session_state.auth_action = None
                st.rerun()
        
        elif st.session_state.auth_action == "login":
            app.authenticate_user()
            # Back button
            if st.button("← Back to Main Menu"):
                st.session_state.auth_action = None
                st.rerun()

if __name__ == "__main__":
    main()