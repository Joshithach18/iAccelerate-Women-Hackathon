#module1: user authentication and data collection
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd
import re
import json
from typing import Optional
import streamlit as st

def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

class DataManager:
    def __init__(self, credentials_file: str, spreadsheet_id: str):
        # Initialize Google Sheets connection
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
        client = gspread.authorize(credentials)
        self.sheet = client.open_by_key(spreadsheet_id).sheet1

    def fetch_user_data(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch user data from Google Sheets"""
        all_data = self.sheet.get_all_records()
        return next((row for row in all_data if row['username'] == username), None)

    def get_user_financial_data(self, username: str) -> Union[Dict[str, Any], pd.DataFrame]:
        """Get financial data for analysis, compatible with both Module 1 and Module 3"""
        try:
            records = self.sheet.get_all_records()
            user_data = []
            for record in records:
                if record['username'] == username:
                    user_data.append(record)
        
            # Convert to DataFrame
            if user_data:
                return pd.DataFrame(user_data)
        
            return pd.DataFrame()  # Return empty DataFrame instead of empty dict
        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")
            return pd.DataFrame()

    def get_user_details_streamlit(self) -> Dict[str, Any]:
        """Get user registration details via Streamlit UI"""
        user_data = {}
        
        # Using st.form to collect all data at once
        with st.form("registration_form"):
            st.subheader("📝 User Registration")
            
            # Basic user information
            user_data["username"] = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email address")
            user_data["password"] = st.text_input("Password", type="password", placeholder="Set a secure password")
            
            # Financial information
            col1, col2 = st.columns(2)
            with col1:
                user_data["monthly_income"] = st.number_input("Monthly Income (₹)", min_value=0.0, step=1000.0)
                user_data["budget_limit"] = st.number_input("Budget Limit for Next Month (₹)", min_value=0.0, step=1000.0)
            
            with col2:
                user_data["savings_goal"] = st.number_input("Savings Goal for Next Month (₹)", min_value=0.0, step=1000.0)
            
            # Spending categories
            st.subheader("💸 Monthly Expenses")
            
            categories = ["Food", "Rent", "Utilities", "Entertainment", "Loan Payments", "Savings", "Others"]
            spent_categories = {}
            
            # Create a 2-column layout for categories
            col1, col2 = st.columns(2)
            
            # Distribute categories across columns
            for i, category in enumerate(categories):
                with col1 if i % 2 == 0 else col2:
                    spent_categories[category] = st.number_input(
                        f"{category} (₹)",
                        min_value=0.0,
                        step=100.0,
                        key=f"cat_{category}"
                    )
            
            # Submit button
            submitted = st.form_submit_button("Register ✨")
        
        # Process form submission
        if submitted:
            # Validate email
            if not is_valid_email(email):
                st.error("📧 Please enter a valid email address.")
                return None
            
            # Check if email already exists
            existing_emails = self.sheet.col_values(2)  # Get all emails in column 2
            if email in existing_emails:
                st.error("📧 This email is already registered. Please log in or use a different email.")
                return None
            
            # Complete user data
            user_data["email"] = email
            user_data["spent_categories"] = spent_categories
            user_data["total_spent"] = sum(spent_categories.values())
            user_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_data["updated_at"] = user_data["created_at"]
            
            return user_data
        
        return None

    def get_user_details(self) -> Dict[str, Any]:
        """
        Original console-based user details function
        Kept for backwards compatibility
        """
        user_data = {}
        print("Welcome to the Women's Financial Empowerment App!")
        user_data["username"] = input("Enter your username: ")
        while True:
            email = input("Enter your email: ")
            if is_valid_email(email):
                existing_users = self.sheet.col_values(2)  # Get all emails in column 2
                if email in existing_users:
                    print("Email already exists. Please log in or use a different email.")
                    return None
                user_data["email"] = email
                break
            else:
                print("Invalid email format. Please enter a valid email.")
        
        import getpass
        user_data["password"] = getpass.getpass("Set a password: ")  # Masked input for password security
        
        while True:
            try:
                user_data["monthly_income"] = float(input("Enter your monthly income: "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value for income.")
        
        print("\nEnter your spending details for different categories (in numbers):")
        categories = ["Food", "Rent", "Utilities", "Entertainment", "Loan Payments", "Savings", "Others"]
        spent_categories = {}
        for category in categories:
            while True:
                try:
                    spent_categories[category] = float(input(f"How much did you spend on {category}?: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        
        user_data["spent_categories"] = spent_categories
        user_data["total_spent"] = sum(spent_categories.values())
        
        while True:
            try:
                user_data["budget_limit"] = float(input("Set a budget limit for next month: "))
                user_data["savings_goal"] = float(input("Enter your savings goal for next month: "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
        
        user_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_data["updated_at"] = user_data["created_at"]
        return user_data
    
    def save_user_data(self, user_data: Dict[str, Any]) -> None:
        """Save user data to Google Sheets"""
        try:
            if not user_data:
                return
                
            spent_categories = user_data["spent_categories"]  # Extracting spent categories
            data = [
                user_data["username"],
                user_data["email"],
                user_data["password"],
                user_data["monthly_income"],
                spent_categories["Food"],
                spent_categories["Rent"], 
                spent_categories["Utilities"], 
                spent_categories["Entertainment"], 
                spent_categories["Loan Payments"], 
                spent_categories["Savings"], 
                spent_categories["Others"],
                user_data["total_spent"], 
                user_data["budget_limit"], 
                user_data["savings_goal"], 
                user_data["created_at"], 
                user_data["updated_at"]
            ]
            self.sheet.append_row(data)
            return True
        except Exception as e:
            st.error(f"Error saving user data: {str(e)}")
            return False
    
    def update_user_data(self, username: str, updated_data: Dict[str, Any]) -> bool:
        """Update existing user data in Google Sheets"""
        try:
            # Find the user's row
            all_data = self.sheet.get_all_records()
            user_row = None
            for i, row in enumerate(all_data):
                if row['username'] == username:
                    user_row = i + 2  # +2 because sheets are 1-indexed and we have header row
                    break
            
            if not user_row:
                return False
            
            # Update data
            cells_to_update = []
            
            # Map column indices to data keys
            column_mapping = {
                3: "monthly_income",  # Assuming column C is monthly_income
                4: "Food",
                5: "Rent",
                6: "Utilities",
                7: "Entertainment",
                8: "Loan Payments",
                9: "Savings",
                10: "Others",
                11: "total_spent",
                12: "budget_limit",
                13: "savings_goal"
            }
            
            # Update each cell as needed
            for col_idx, key in column_mapping.items():
                if key in updated_data or (key in ["Food", "Rent", "Utilities", "Entertainment", "Loan Payments", "Savings", "Others"] and "spent_categories" in updated_data):
                    value = updated_data[key] if key in updated_data else updated_data["spent_categories"][key]
                    cell = self.sheet.cell(user_row, col_idx)
                    cell.value = value
                    cells_to_update.append(cell)
            
            # Update timestamp
            updated_at_cell = self.sheet.cell(user_row, 16)  # Column P
            updated_at_cell.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cells_to_update.append(updated_at_cell)
            
            # Batch update to minimize API calls
            self.sheet.update_cells(cells_to_update)
            return True
            
        except Exception as e:
            st.error(f"Error updating user data: {str(e)}")
            return False
#module2: Budget tracking and expense tracking
import streamlit as st
import getpass
from typing import Dict, Any, Optional
import json
import cohere
import math
import matplotlib.pyplot as plt
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px

class FinancialVisualizer:
    def format_money(self, amount: float) -> str:
        """Format money values consistently"""
        return f"${amount:,.2f}"

    def plot_spending_distribution(self, user_data: Dict[str, Any]):
        """Pie chart for spending distribution using Plotly"""
        categories = user_data.get('categories', {})
        if not categories:
            return go.Figure()  # Return empty figure if no categories exist

        labels = list(categories.keys())
        values = list(categories.values())

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hoverinfo='label+percent',
                textinfo='value',
                textfont_size=14,
                marker=dict(colors=px.colors.qualitative.Set3, line=dict(color='#000000', width=2))
            )
        )
        fig.update_layout(title_text="📊 Spending Distribution")
        return fig

    def plot_budget_vs_actual(self, user_data: Dict[str, Any]):
        """Bar chart for budget vs actual spending using Plotly"""
        categories = user_data.get('categories', {})
        budget_limits = user_data.get('budget_limits', {})
        if not categories:
            return go.Figure()

        labels = list(categories.keys())
        actual_spending = list(categories.values())
        budgeted_spending = [budget_limits.get(cat, 0) for cat in labels]  # Get per-category budget

        fig = go.Figure(data=[
            go.Bar(name='Budget', x=labels, y=budgeted_spending, marker_color='skyblue'),
            go.Bar(name='Actual', x=labels, y=actual_spending, marker_color='orange')
        ])
        fig.update_layout(
            barmode='group',
            title='📈 Budget vs Actual',
            xaxis_title='Categories',
            yaxis_title='Amount Spent',
            xaxis_tickangle=-45,
            legend_title='Spending',
            template='plotly'
        )
        return fig

    def plot_savings_progress(self, user_data: Dict[str, Any]):
        """Bar chart for savings progress using Plotly"""
        monthly_income = user_data.get('monthly_income', 0)
        total_spent = user_data.get('total_spent', 0)
        previous_savings = user_data.get('previous_savings', 0)
        savings_goal = user_data.get('savings_goal', 0)
        
        current_savings = max(previous_savings + (monthly_income - total_spent), 0)
        
        labels = ['Current Savings', 'Savings Goal']
        values = [current_savings, savings_goal]
        colors = ['green', 'grey']
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels, 
                y=values, 
                marker_color=colors,
                text=[self.format_money(value) for value in values], 
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='💰 Savings Progress',
            yaxis_title='Amount',
            template='plotly'
        )
        return fig

    def create_visualizations(self, user_data: Dict[str, Any]):
        """Create all visualizations for Streamlit"""
        return {
            "spending_distribution": self.plot_spending_distribution(user_data),
            "budget_vs_actual": self.plot_budget_vs_actual(user_data),
            "savings_progress": self.plot_savings_progress(user_data)
        }

class FinancialAnalyzer:
    def __init__(self, cohere_api_key: str):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.visualizer = FinancialVisualizer()

    def _calculate_metrics(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial metrics"""
        monthly_income = user_data.get('monthly_income', 1)  # Avoid division by zero
        total_spent = user_data.get('total_spent', 0)
        budget_limit = user_data.get('budget_limit', 0)
        savings_goal = user_data.get('savings_goal', 0)
        previous_savings = user_data.get('previous_savings', 0)

        return {
            'monthly_income': monthly_income,
            'total_spent': total_spent,
            'spending_ratio': (total_spent / monthly_income) * 100,
            'savings_gap': savings_goal - (previous_savings + (monthly_income - total_spent)),
            'budget_variance': total_spent - budget_limit
        }

    def _generate_analysis_prompt(self, metrics: Dict[str, float]) -> str:
        """Generate analysis prompt"""
        return f"""As a financial advisor, analyze this financial situation:
        Monthly Income: ${metrics['monthly_income']:,.2f}
        Total Expenses: ${metrics['total_spent']:,.2f}
        Spending Ratio: {metrics['spending_ratio']:.1f}%
        Budget Variance: ${metrics['budget_variance']:,.2f}
        Savings Gap: ${metrics['savings_gap']:,.2f}
        Provide:
        1. 💰 Spending Analysis
        2. 💎 Savings Assessment
        3. 📊 Budget Optimization
        4. ⚠️ Risk Assessment
        Use emojis and clear sections with actionable recommendations."""

    def get_financial_insights(self, data_key: str, user_data_str: str) -> tuple:
        """Generate combined financial insights and visualizations"""
        user_data = json.loads(user_data_str)
        metrics = self._calculate_metrics(user_data)
        
        # Get AI-generated insights
        prompt = self._generate_analysis_prompt(metrics)
        response = self.cohere_client.chat(
            model='command-xlarge-nightly',
            message=prompt,
            max_tokens=1000,
            temperature=0.7,
        )
        ai_insights = response.text
        
        # Get visualizations
        visualizations = self.visualizer.create_visualizations(user_data)
        
        return ai_insights, visualizations, user_data
#module3: Financial planning and goal setting
import json
import getpass
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import cohere
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime, timedelta
import numpy as np

# --- Module 3 Classes ---
class Module3Visualizer:
    def format_money(self, amount: float) -> str:
        return f"₹{amount:,.2f}"

    def plot_budget_vs_actual(self, user_data):
        """Create budget vs actual plot using Plotly for Streamlit"""
        categories = user_data['categories']
        labels = list(categories.keys())
        actual_spending = list(categories.values())
        budget_per_category = user_data['budget_limit'] / len(categories)
        budgeted_spending = [budget_per_category] * len(categories)

        fig = go.Figure(data=[
            go.Bar(name='Budget', x=labels, y=budgeted_spending, marker_color='skyblue'),
            go.Bar(name='Actual', x=labels, y=actual_spending, marker_color='orange')
        ])
        
        fig.update_layout(
            barmode='group',
            title='📈 Budget vs Actual',
            xaxis_title='Categories',
            yaxis_title='Amount Spent',
            xaxis_tickangle=-45,
            legend_title='Spending',
            template='plotly'
        )
        
        return fig

class Module3Analyzer:
    def __init__(self, cohere_api_key: str):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.visualizer = Module3Visualizer()

    def ai_powered_insights(self, metrics):
        prompt = f"""Analyze the following financial metrics: {metrics}"""
        response = self.cohere_client.chat(
            model='command-xlarge-nightly',
            message=prompt,
            max_tokens=600,
            temperature=0.7
        )
        return response.text

    def time_series_forecasting(self, df):
        """
        Perform time series forecasting with synthetic historical data generation
        for cases with limited data points and return Plotly figure for Streamlit
        """
        try:
            # Get the latest data point
            latest_data = df.iloc[-1]
            current_date = pd.to_datetime(latest_data['created_at'])
        
            # Create synthetic historical data for the past 6 months
            dates = pd.date_range(end=current_date, periods=6, freq='M')
        
            # Create historical trend based on current savings
            current_savings = float(latest_data['Savings'])
            # Assume slight variations in past months (90-98% of current value)
            historical_savings = [
                current_savings * factor for factor in [0.90, 0.92, 0.94, 0.96, 0.98, 1.0]
            ]
        
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': dates,
                'y': historical_savings
            })
        
            # Create and fit the Prophet model
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95
            )
            model.fit(forecast_df)
        
            # Create future dates for forecasting next 6 months
            future = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future)
            
            # Create Plotly figure instead of matplotlib
            fig = go.Figure()
            
            # Add historical data points
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['y'],
                mode='markers+lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='rgba(31, 119, 180, 0.8)')
            ))
            
            # Add uncertainty intervals
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'].iloc[::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'].iloc[::-1]]),
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo='skip',
                showlegend=False
            ))
            
            fig.update_layout(
                title='Savings Forecast (Next 6 Months)',
                xaxis_title='Date',
                yaxis_title='Savings Amount (₹)',
                hovermode='x unified',
                template='plotly'
            )
        
            # Calculate insights
            latest_value = historical_savings[-1]
            forecast_value = forecast['yhat'].iloc[-1]
            growth = ((forecast_value - latest_value) / latest_value) * 100
            
            # Monthly targets
            monthly_targets = forecast['yhat'].tail(6)
            target_dates = forecast['ds'].tail(6)
            monthly_goals = {}
            
            for i in range(len(monthly_targets)):
                month = target_dates.iloc[i].strftime('%B %Y')
                monthly_goals[month] = monthly_targets.iloc[i]
            
            forecast_insights = {
                "current_savings": latest_value,
                "forecasted_savings": forecast_value,
                "growth_rate": growth,
                "monthly_targets": monthly_goals
            }
            
            return fig, forecast_insights
        
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
            st.warning("Troubleshooting tips:")
            st.warning("1. Ensure your savings data is entered as numbers")
            st.warning("2. Check that your dates are properly formatted")
            st.warning("3. Verify that all required data is present")
            return None, None

    def analyze_and_visualize(self, user_data):
        """Analyze and visualize financial data for Streamlit"""
        try:
            budget_vs_actual_fig = self.visualizer.plot_budget_vs_actual(user_data)
            insights = self.ai_powered_insights(user_data)
            return budget_vs_actual_fig, insights
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.warning("Please ensure your financial data is complete and properly formatted.")
            return None, None

class Module3App:
    def __init__(self, cohere_api_key: str):
        # Initialize the analyzer with the API key
        self.analyzer = Module3Analyzer(cohere_api_key)

    def get_user_data(self, df):
        user_data = {
            'monthly_income': df['monthly_income'].iloc[-1],
            'categories': {
                'Food': df['Food'].iloc[-1],
                'Rent': df['Rent'].iloc[-1],
                'Utilities': df['Utilities'].iloc[-1],
                'Entertainment': df['Entertainment'].iloc[-1],
                'Loan Payments': df['Loan Payments'].iloc[-1],
                'Savings': df['Savings'].iloc[-1],
                'Others': df['Others'].iloc[-1]
            },
            'budget_limit': df['budget_limit'].iloc[-1]
        }
        return user_data

    def show_financial_planning(self, user_data):
        # In Streamlit version, this will be displayed in the UI
        return {
            "monthly_income": user_data['monthly_income'],
            "categories": user_data['categories']
        }

    def run_module3(self, df, api_key):
        """
        Run Module 3 for financial planning and analysis in Streamlit
        using the provided DataFrame
        """
        st.title("📊 Financial Planning & Analysis")
        
        if df.empty:
            st.error("No data found in the spreadsheet.")
            return
        
        # Initialize data
        user_data = self.get_user_data(df)
        
        # Display financial overview
        st.header("💰 Financial Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Monthly Income", f"₹{user_data['monthly_income']:,.2f}")
            st.metric("Budget Limit", f"₹{user_data['budget_limit']:,.2f}")
        with col2:
            total_spent = sum(user_data['categories'].values()) - user_data['categories']['Savings']
            st.metric("Total Expenses", f"₹{total_spent:,.2f}")
            st.metric("Savings", f"₹{user_data['categories']['Savings']:,.2f}")
        
        # Category breakdown 
        st.subheader("Category Breakdown")
        category_df = pd.DataFrame({
            'Category': list(user_data['categories'].keys()),
            'Amount': list(user_data['categories'].values())
        })
        
        # Add interactive element for category view
        view_option = st.radio("View as:", ["Table", "Bar Chart", "Pie Chart"], horizontal=True)
        
        if view_option == "Table":
            st.dataframe(category_df, hide_index=True)
        elif view_option == "Bar Chart":
            fig = px.bar(category_df, x='Category', y='Amount', color='Category')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.pie(category_df, values='Amount', names='Category')
            st.plotly_chart(fig, use_container_width=True)
        
        # Budget Analysis
        st.header("📈 Budget Analysis")
        budget_vs_actual_fig, insights = self.analyzer.analyze_and_visualize(user_data)
        
        if budget_vs_actual_fig:
            st.plotly_chart(budget_vs_actual_fig, use_container_width=True)
        
        # AI Insights
        st.header("🤖 AI-Powered Financial Insights")
        if insights:
            st.markdown(insights)
        
        # Forecasting
        st.header("🔮 Savings Forecast")
        
        # Add some interactive options for forecasting
        forecast_period = st.slider("Forecast Period (Months)", min_value=3, max_value=12, value=6)
        
        # Adjust DataFrame for the selected period if needed
        # Here we're just passing the original df, but you could modify based on the slider
        
        # Generate forecast
        forecast_fig, forecast_insights = self.analyzer.time_series_forecasting(df)
        
        if forecast_fig and forecast_insights:
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Display forecast insights
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Savings", 
                    f"₹{forecast_insights['current_savings']:,.2f}"
                )
            with col2:
                st.metric(
                    "Forecasted Savings (6 months)", 
                    f"₹{forecast_insights['forecasted_savings']:,.2f}",
                    delta=f"{forecast_insights['growth_rate']:.1f}%"
                )
            
            # Display growth insights
            st.subheader("Growth Analysis")
            if forecast_insights['growth_rate'] > 0:
                st.success("📈 Positive Growth Trajectory")
                st.markdown("""
                - Your savings are projected to grow steadily
                - Keep maintaining your current savings rate
                - Consider increasing your monthly savings if possible
                """)
            else:
                st.warning("📉 Areas for Improvement")
                st.markdown("""
                - Consider reviewing your monthly expenses
                - Look for opportunities to increase your savings rate
                - Set up automatic transfers to your savings account
                """)
            
            # Display monthly goals in a nice table
            st.subheader("📊 Monthly Savings Targets")
            
            monthly_goals_df = pd.DataFrame({
                'Month': list(forecast_insights['monthly_targets'].keys()),
                'Target': list(forecast_insights['monthly_targets'].values())
            })
            
            st.dataframe(
                monthly_goals_df.style.format({'Target': '₹{:,.2f}'}),
                hide_index=True
            )
        
        # Tips and recommendations section
        with st.expander("💡 Tips & Recommendations"):
            st.markdown("""
            ### Financial Health Tips
            1. **50/30/20 Rule**: Try to allocate 50% of income to needs, 30% to wants, and 20% to savings
            2. **Emergency Fund**: Aim to save 3-6 months of expenses in an emergency fund
            3. **Debt Management**: Prioritize paying off high-interest debt
            4. **Automate Savings**: Set up automatic transfers on payday
            5. **Review and Adjust**: Regularly review your budget and make adjustments
            """)
#module4: SIP & EMI calculator
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Module4App:
    def __init__(self):
        pass
        
    def sip_calculator(self):
        st.subheader("📈 SIP Calculator - Estimate your Future Investment Value!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            P = st.number_input("💰 Monthly Investment Amount (₹)", min_value=100.0, value=5000.0, step=100.0)
            
        with col2:
            t = st.slider("📅 Investment Duration (years)", min_value=1, max_value=30, value=10)
            
        r = st.slider("📊 Expected Rate of Return (Annual %)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
        
        # Convert annual rate to decimal
        r = r / 100
        # Monthly compounding
        n = 12
        # Calculate maturity value
        maturity = P * (((1 + r/n)**(n*t) - 1) / (r/n)) * (1 + r/n)
        
        # Calculate invested amount
        invested_amount = P * 12 * t
        
        # Calculate wealth gained
        wealth_gained = maturity - invested_amount
        
        # Display results in a nicer format
        st.markdown("### 🎯 Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Invested", f"₹{invested_amount:,.2f}")
        col2.metric("Wealth Gained", f"₹{wealth_gained:,.2f}")
        col3.metric("Final Corpus", f"₹{maturity:,.2f}")
        
        # Generate data for chart
        years = list(range(1, t+1))
        corpus_values = []
        invested_values = []
        
        for year in years:
            invested = P * 12 * year
            invested_values.append(invested)
            
            corpus = P * (((1 + r/n)**(n*year) - 1) / (r/n)) * (1 + r/n)
            corpus_values.append(corpus)
        
        # Create DataFrame for chart
        chart_data = pd.DataFrame({
            'Year': years,
            'Invested Amount': invested_values,
            'Corpus Value': corpus_values
        })
        
        # Plot the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(chart_data['Year'], chart_data['Invested Amount'], label='Invested Amount', color='lightblue')
        ax.bar(chart_data['Year'], chart_data['Corpus Value'] - chart_data['Invested Amount'], 
               bottom=chart_data['Invested Amount'], label='Wealth Gained', color='lightgreen')
        
        ax.set_xlabel('Years')
        ax.set_ylabel('Amount (₹)')
        ax.set_title('SIP Growth Over Time')
        ax.legend()
        
        # Draw gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis labels to show with commas
        import matplotlib.ticker as mtick
        fmt = '{x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)
        
        # Rotate x-axis labels for better readability
        plt.xticks(years[::max(1, t//10)])
        
        st.pyplot(fig)
        
        # Add explanation
        with st.expander("ℹ️ How SIP works?"):
            st.markdown("""
            ### Understanding SIP (Systematic Investment Plan)
            
            A Systematic Investment Plan (SIP) allows you to invest a fixed amount regularly in mutual funds. The benefits include:
            
            - **Rupee Cost Averaging**: You buy more units when prices are low and fewer units when prices are high
            - **Power of Compounding**: As your investment grows, the returns generate additional returns
            - **Disciplined Investing**: Regular investments create a habit of saving
            - **Flexibility**: You can start with small amounts and increase over time
            
            The formula used for calculation is:
            
            M = P × (((1 + r/n)^(n×t) - 1) / (r/n)) × (1 + r/n)
            
            Where:
            - M = Maturity Amount
            - P = Monthly Investment
            - r = Annual Rate of Return (decimal)
            - t = Investment Duration in Years
            - n = Number of Compounding Periods per Year (12 for monthly)
            """)

    def emi_calculator(self):
        st.subheader("🏠 EMI Calculator - Calculate Your Monthly Loan Installment!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            P = st.number_input("💵 Principal Loan Amount (₹)", min_value=1000.0, value=1000000.0, step=10000.0)
            
        with col2:
            tenure_years = st.slider("📅 Loan Tenure (years)", min_value=1, max_value=30, value=20)
            
        annual_rate = st.slider("📊 Interest Rate (Annual %)", min_value=1.0, max_value=20.0, value=8.5, step=0.1)
        
        # Convert annual rate to monthly and decimal
        r = (annual_rate / 100) / 12
        # Total number of monthly payments
        n = tenure_years * 12
        # Calculate EMI
        emi = P * r * ((1 + r)**n) / (((1 + r)**n) - 1)
        
        # Calculate total payment and interest
        total_payment = emi * n
        total_interest = total_payment - P
        
        # Display results
        st.markdown("### 📊 Loan Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly EMI", f"₹{emi:,.2f}")
        col2.metric("Total Interest", f"₹{total_interest:,.2f}")
        col3.metric("Total Payment", f"₹{total_payment:,.2f}")
        
        # Generate amortization schedule
        remaining_principal = P
        yearly_data = []
        
        interest_paid_yearly = 0
        principal_paid_yearly = 0
        
        for payment_no in range(1, n+1):
            interest_payment = remaining_principal * r
            principal_payment = emi - interest_payment
            remaining_principal -= principal_payment
            
            interest_paid_yearly += interest_payment
            principal_paid_yearly += principal_payment
            
            if payment_no % 12 == 0 or payment_no == n:
                year = (payment_no - 1) // 12 + 1
                yearly_data.append({
                    'Year': year,
                    'Principal Paid': principal_paid_yearly,
                    'Interest Paid': interest_paid_yearly,
                    'Remaining Principal': max(0, remaining_principal)
                })
                interest_paid_yearly = 0
                principal_paid_yearly = 0
        
        # Create DataFrame for yearly data
        yearly_df = pd.DataFrame(yearly_data)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(yearly_df['Year'], yearly_df['Principal Paid'], label='Principal Paid', color='lightgreen')
        ax.bar(yearly_df['Year'], yearly_df['Interest Paid'], bottom=yearly_df['Principal Paid'], 
               label='Interest Paid', color='salmon')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Amount (₹)')
        ax.set_title('Yearly Payment Breakdown')
        ax.legend()
        
        # Format y-axis labels to show with commas
        import matplotlib.ticker as mtick
        fmt = '{x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(yearly_df['Year'][::max(1, len(yearly_df)//10)])
        
        st.pyplot(fig)
        
        # Amortization schedule table
        with st.expander("📋 View Yearly Amortization Schedule"):
            st.dataframe(yearly_df.style.format({
                'Principal Paid': '₹{:,.2f}',
                'Interest Paid': '₹{:,.2f}',
                'Remaining Principal': '₹{:,.2f}'
            }))
        
        # Add explanation
        with st.expander("ℹ️ How EMI works?"):
            st.markdown("""
            ### Understanding EMI (Equated Monthly Installment)
            
            An EMI is a fixed payment amount made by a borrower to a lender at a specified date each month. EMIs consist of both principal and interest components.
            
            The formula used for calculation is:
            
            EMI = P × r × (1 + r)^n / ((1 + r)^n - 1)
            
            Where:
            - P = Principal Loan Amount
            - r = Monthly Interest Rate (Annual Rate ÷ 12 ÷ 100)
            - n = Total Number of Monthly Payments (Tenure in Years × 12)
            
            In the early years of the loan, a larger portion of each EMI payment goes toward interest, while in later years, more goes toward reducing the principal amount.
            """)

    def run_module4(self):
        st.title("💡 SIP & EMI Calculators 💡")
        
        tab1, tab2 = st.tabs(["SIP Calculator", "EMI Calculator"])
        
        with tab1:
            self.sip_calculator()
            
        with tab2:
            self.emi_calculator()
#module 5: Investment portfolio tracker
import pandas as pd
import numpy as np
import cohere
import plotly.graph_objects as go
import plotly.express as px
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import streamlit as st
from datetime import datetime
import json
import os

class Module5Investment:
    def __init__(self, credentials_file=None, investment_spreadsheet_id=None, cohere_api_key=None):
        self.credentials_file = credentials_file
        self.investment_spreadsheet_id = investment_spreadsheet_id
        self.cohere_api_key = cohere_api_key
        
        # Initialize cohere if API key is provided
        if cohere_api_key:
            self.cohere = cohere.Client(cohere_api_key)
        else:
            self.cohere = None
        
        # Initialize Google Sheets if credentials and spreadsheet ID are provided
        self.service = self.init_google_sheets()
        
        # Load stock dataset
        self.dataset_path = "stockmarketdata.csv"
        self.stock_data = self.load_stock_data()
        
        # Initialize or load investment data
        self.investment_data = self.load_investment_data()

    def load_investment_data(self):
        """Load investment data from Google Sheets or session state."""
        if self.service and self.investment_spreadsheet_id:
            try:
                # Fetch data from Google Sheets
                sheet = self.service.spreadsheets()
                result = sheet.values().get(spreadsheetId=self.investment_spreadsheet_id,
                                             range="A1:E").execute()
                values = result.get('values', [])
                
                if not values:
                    st.warning("No investment data found in Google Sheet.")
                    return pd.DataFrame(
                        columns=["Username", "Stock Name", "Purchase Price", "Quantity", "Date"]
                    )
                
                # Convert to DataFrame
                column_names = values[0]  # Header row
                data = values[1:]
                df = pd.DataFrame(data, columns=column_names)
                
                # Ensure correct data types
                df["Purchase Price"] = pd.to_numeric(df["Purchase Price"], errors='coerce')
                df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                
                return df
                
            except Exception as e:
                st.error(f"Error loading investment data from Google Sheets: {e}")
                st.info("Falling back to session state data.")
                if 'investment_data' not in st.session_state:
                    st.session_state.investment_data = pd.DataFrame(
                        columns=["Username", "Stock Name", "Purchase Price", "Quantity", "Date"]
                    )
                return st.session_state.investment_data
                
        else:
            st.info("Google Sheets not initialized. Loading investment data from session state.")
            if 'investment_data' not in st.session_state:
                st.session_state.investment_data = pd.DataFrame(
                    columns=["Username", "Stock Name", "Purchase Price", "Quantity", "Date"]
                )
            return st.session_state.investment_data

    def load_stock_data(self):
        try:
            # Check if data is already in session state
            if 'stock_data' in st.session_state:
                return st.session_state.stock_data
                
            # Try to load from file
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                df = df.dropna(subset=["Price", "Market Cap", "Earnings/Share", "Price/Earnings"])
                st.session_state.stock_data = df
                return df
            else:
                # Use sample data if file doesn't exist (for demo purposes)
                sample_data = {
                    "Symbol": ["RELIANCE", "HDFC", "TCS", "INFY", "ITC", "SBIN", "ICICI", "LT", "BHARTI", "WIPRO"],
                    "Name": ["Reliance Industries", "HDFC Bank", "Tata Consultancy", "Infosys", "ITC Limited", 
                             "State Bank of India", "ICICI Bank", "Larsen & Toubro", "Bharti Airtel", "Wipro"],
                    "Sector": ["Energy", "Banking", "IT", "IT", "FMCG", "Banking", "Banking", "Construction", "Telecom", "IT"],
                    "Price": [2750.25, 1650.75, 3425.50, 1525.25, 450.30, 625.45, 950.20, 2350.75, 875.30, 425.65],
                    "Market Cap": ["16.5T", "9.2T", "12.5T", "6.3T", "5.6T", "5.5T", "6.6T", "3.3T", "4.9T", "2.3T"],
                    "Earnings/Share": [115.5, 72.3, 88.2, 53.7, 14.6, 42.5, 30.4, 65.8, 12.5, 19.2],
                    "Price/Earnings": [23.8, 22.8, 38.8, 28.4, 30.8, 14.7, 31.3, 35.7, 70.0, 22.2],
                    "Dividend Yield": [0.5, 0.7, 1.2, 2.3, 4.1, 1.3, 0.5, 0.9, 0.4, 1.1],
                    "52 Week High": [2950.50, 1780.25, 3680.75, 1625.40, 485.90, 660.30, 1020.45, 2560.80, 960.25, 450.35],
                    "52 Week Low": [2340.75, 1450.30, 3120.60, 1380.25, 350.15, 560.40, 840.55, 2120.30, 760.45, 380.20]
                }
                df = pd.DataFrame(sample_data)
                st.session_state.stock_data = df
                return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None

    def init_google_sheets(self):
        """Initialize Google Sheets API service."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file) and self.investment_spreadsheet_id:
                creds = Credentials.from_service_account_file(
                    self.credentials_file, 
                    scopes=["https://www.googleapis.com/auth/spreadsheets"]
                )
                return build("sheets", "v4", credentials=creds)
            else:
                st.warning("Credentials file or investment spreadsheet ID is missing.")
                return None
        except Exception as e:
            st.error(f"Could not initialize Google Sheets: {e}")
            st.info("Investment data will be saved to session state only.")
            return None

    def get_stock_insights(self, symbol):
        if self.stock_data is None:
            return "Stock data not available."
        stock = self.stock_data[self.stock_data["Symbol"] == symbol]
        if stock.empty:
            return "Stock symbol not found."
        insights = {
            "Name": stock["Name"].values[0],
            "Sector": stock["Sector"].values[0],
            "Price": stock["Price"].values[0],
            "52 Week High": stock["52 Week High"].values[0],
            "52 Week Low": stock["52 Week Low"].values[0],
            "Market Cap": stock["Market Cap"].values[0],
            "P/E Ratio": stock["Price/Earnings"].values[0],
            "Dividend Yield": stock["Dividend Yield"].values[0] if not np.isnan(stock["Dividend Yield"].values[0]) else "N/A"
        }
        return insights

    def analyze_stock_performance(self, symbol):
        stock = self.get_stock_insights(symbol)
        if isinstance(stock, str):
            return stock
        
        message = f"Stock: {stock['Name']} ({symbol})\n"
        message += f"Sector: {stock['Sector']}\n"
        message += f"Current Price:  ₹{stock['Price']}\n"
        message += f"52 Week Range:  ₹{stock['52 Week Low']} -  ₹{stock['52 Week High']}\n"
        message += f"Market Cap: {stock['Market Cap']}\n"
        message += f"P/E Ratio: {stock['P/E Ratio']}\n"
        message += f"Dividend Yield: {stock['Dividend Yield']}\n"
        
        return message

    def suggest_investment_opportunities(self):
        strong_stocks = self.stock_data[(self.stock_data["Price/Earnings"] < 20) & (self.stock_data["Earnings/Share"] > 5)]
        return strong_stocks[["Symbol", "Name", "Sector", "Price", "Earnings/Share", "Price/Earnings"]].head(10)

    def get_investment_data(self, username):
        investment_data = self.load_investment_data()  # Ensure investment data is loaded
        return investment_data[investment_data["Username"] == username]

    def save_investment_data(self, username, stock, price, quantity, date):
        """Save new investment to Google Sheets and local DataFrame."""
        new_data = pd.DataFrame([[username, stock, price, quantity, date]], 
                                columns=["Username", "Stock Name", "Purchase Price", "Quantity", "Date"])
        
        # Update investment data
        if self.service and self.investment_spreadsheet_id:
            try:
                new_row = [[username, stock, price, quantity, date]]
                self.service.spreadsheets().values().append(
                    spreadsheetId=self.investment_spreadsheet_id,
                    range="A1",
                    valueInputOption="USER_ENTERED",
                    body={"values": new_row}
                ).execute()
                st.success("Investment data saved to Google Sheets!")
            except Exception as e:
                st.error(f"Error saving to Google Sheets: {e}")
                st.info("Saving investment data to session state instead.")
                self.investment_data = pd.concat([self.investment_data, new_data], ignore_index=True)
                st.session_state.investment_data = self.investment_data
        else:
            self.investment_data = pd.concat([self.investment_data, new_data], ignore_index=True)
            st.session_state.investment_data = self.investment_data

    def calculate_returns(self, df):
        if df.empty:
            return df
            
        result_df = df.copy()
        
        # Create a function to safely get stock price
        def get_safe_price(stock_name):
            insights = self.get_stock_insights(stock_name)
            if isinstance(insights, dict):
                return insights["Price"]
            return 0
        
        # Apply calculations
        result_df["Current Price"] = result_df["Stock Name"].apply(get_safe_price)
        result_df["Current Value"] = result_df["Current Price"].astype(float) * result_df["Quantity"].astype(int)
        result_df["Total Invested"] = result_df["Purchase Price"].astype(float) * result_df["Quantity"].astype(int)
        result_df["Returns"] = result_df["Current Value"] - result_df["Total Invested"]
        result_df["Returns (%)"] = (result_df["Returns"] / result_df["Total Invested"]) * 100
        
        return result_df

    def visualize_portfolio(self, df):
        if df.empty:
            return None
            
        # Create pie chart for portfolio allocation
        fig = px.pie(
            df, 
            values="Current Value", 
            names="Stock Name",
            title="📊 Portfolio Allocation",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Update layout for better appearance
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=60, b=60, l=20, r=20)
        )
        
        return fig

    def visualize_returns(self, df):
        if df.empty:
            return None
            
        # Create horizontal bar chart for returns
        fig = px.bar(
            df,
            y="Stock Name",
            x="Returns",
            color="Returns",
            labels={"Returns": "Returns (₹)", "Stock Name": "Stock"},
            title="📈 Returns by Stock",
            orientation='h',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Returns (₹)",
            yaxis_title="Stock",
            margin=dict(t=60, b=40, l=80, r=20)
        )
        
        return fig

    def ai_recommendations(self, df):
        if self.cohere is None:
            return "AI recommendations require a Cohere API key."
            
        if df.empty:
            return "No investment data available for AI analysis."
            
        try:
            total_invested = df["Total Invested"].sum()
            total_value = df["Current Value"].sum()
            returns_pct = ((total_value - total_invested) / total_invested) * 100
            
            prompt = f"""
            My investment portfolio:
            - Total invested: ₹{total_invested:,.2f}
            - Current value: ₹{total_value:,.2f}
            - Overall returns: {returns_pct:.2f}%
            
            Provide concise investment insights and recommendations based on my current portfolio.
            Focus on diversification, risk management, and potential opportunities.
            Keep your response under 300 words.
            """
            
            response = self.cohere.chat(
                model="command-xlarge-nightly", 
                message=prompt,
                max_tokens=300
            )
            
            return response.text
        except Exception as e:
            return f"Error generating AI recommendations: {e}"

    def run_module5(self):
        st.title("📊 Investment Portfolio Tracker")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Portfolio Dashboard", 
            "Add Investment", 
            "Stock Analysis", 
            "Investment Recommendations"
        ])
        
        # Get username (simplified for demo)
        if 'username' not in st.session_state:
            st.session_state.username = "DefaultUser"
        username = st.session_state.username
        
        # Tab 1: Portfolio Dashboard
        with tab1:
            st.subheader("Your Investment Portfolio")
            
            # Get user's investment data
            user_investments = self.get_investment_data(username)
            
            if user_investments.empty:
                st.info("📝 You haven't added any investments yet. Go to the 'Add Investment' tab to get started.")
            else:
                # Calculate returns
                portfolio_data = self.calculate_returns(user_investments)
                
                # Summary metrics
                total_invested = portfolio_data["Total Invested"].sum()
                current_value = portfolio_data["Current Value"].sum()
                total_returns = current_value - total_invested
                returns_pct = (total_returns / total_invested) * 100 if total_invested > 0 else 0
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Invested", f"₹{total_invested:,.2f}")
                col2.metric("Current Value", f"₹{current_value:,.2f}")
                col3.metric("Total Returns", f"₹{total_returns:,.2f}", 
                           f"{returns_pct:+.2f}%")
                
                # Count of stocks
                stock_count = portfolio_data["Stock Name"].nunique()
                col4.metric("Stocks", f"{stock_count}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Portfolio allocation chart
                    allocation_fig = self.visualize_portfolio(portfolio_data)
                    if allocation_fig:
                        st.plotly_chart(allocation_fig, use_container_width=True)
                
                with col2:
                    # Returns chart
                    returns_fig = self.visualize_returns(portfolio_data)
                    if returns_fig:
                        st.plotly_chart(returns_fig, use_container_width=True)
                
                # Display detailed portfolio table
                st.subheader("Portfolio Details")
                display_cols = ["Stock Name", "Purchase Price", "Current Price", "Quantity", 
                               "Total Invested", "Current Value", "Returns", "Returns (%)", "Date"]
                
                # Format the dataframe for display
                formatted_df = portfolio_data[display_cols].copy()
                formatted_df["Purchase Price"] = formatted_df["Purchase Price"].apply(lambda x: f"₹{float(x):,.2f}")
                formatted_df["Current Price"] = formatted_df["Current Price"].apply(lambda x: f"₹{float(x):,.2f}")
                formatted_df["Total Invested"] = formatted_df["Total Invested"].apply(lambda x: f"₹{float(x):,.2f}")
                formatted_df["Current Value"] = formatted_df["Current Value"].apply(lambda x: f"₹{float(x):,.2f}")
                formatted_df["Returns"] = formatted_df["Returns"].apply(lambda x: f"₹{float(x):,.2f}")
                formatted_df["Returns (%)"] = formatted_df["Returns (%)"].apply(lambda x: f"{float(x):+.2f}%")
                
                st.dataframe(formatted_df, use_container_width=True)
        
        # Tab 2: Add Investment
        with tab2:
            st.subheader("Add New Investment")
            
            # Display available stocks for selection
            if self.stock_data is not None:
                stock_options = self.stock_data["Symbol"].tolist()
                stock_dict = dict(zip(self.stock_data["Symbol"], self.stock_data["Name"]))
                
                # Create stock selection with search
                selected_stock = st.selectbox(
                    "📈 Select Stock",
                    options=stock_options,
                    format_func=lambda x: f"{x} - {stock_dict.get(x, '')}"
                )
                
                if selected_stock:
                    # Get stock insights to display current price
                    insights = self.get_stock_insights(selected_stock)
                    
                    if isinstance(insights, dict):
                        st.info(f"Current Price: ₹{insights['Price']:,.2f}")
                        
                        # Form for investment details
                        with st.form("investment_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                purchase_price = st.number_input(
                                    "💰 Purchase Price (₹)",
                                    min_value=0.01,
                                    value=float(insights['Price']),
                                    step=0.01
                                )
                                
                                quantity = st.number_input(
                                    "🔢 Quantity",
                                    min_value=1,
                                    value=10,
                                    step=1
                                )
                            
                            with col2:
                                purchase_date = st.date_input(
                                    "📅 Purchase Date",
                                    datetime.now().date()
                                )
                                
                                # Calculate total investment
                                total = purchase_price * quantity
                                st.metric("Total Investment", f"₹{total:,.2f}")
                            
                            submit_button = st.form_submit_button("Save Investment")
                            
                            if submit_button:
                                date_str = purchase_date.strftime("%Y-%m-%d")
                                result = self.save_investment_data(
                                    username, selected_stock, purchase_price, 
                                    quantity, date_str
                                )
                                st.success(result)
                    else:
                        st.error(insights)
            else:
                st.error("Stock data not available. Please check if the dataset is loaded correctly.")
        
        # Tab 3: Stock Analysis
        with tab3:
            st.subheader("Stock Analysis")
            
            if self.stock_data is not None:
                # Stock search and analysis
                stock_options = self.stock_data["Symbol"].tolist()
                stock_dict = dict(zip(self.stock_data["Symbol"], self.stock_data["Name"]))
                
                selected_stock = st.selectbox(
                    "🔍 Search for a stock",
                    options=stock_options,
                    format_func=lambda x: f"{x} - {stock_dict.get(x, '')}",
                    key="analysis_stock_select"
                )
                
                if selected_stock:
                    analysis = self.analyze_stock_performance(selected_stock)
                    
                    # Create a card-like display for stock info
                    insights = self.get_stock_insights(selected_stock)
                    
                    if isinstance(insights, dict):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Company info section
                            st.markdown(f"### {insights['Name']} ({selected_stock})")
                            st.markdown(f"**Sector:** {insights['Sector']}")
                            st.markdown(f"**Market Cap:** {insights['Market Cap']}")
                        
                        with col2:
                            # Price metrics in columns
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            
                            metrics_col1.metric(
                                "Current Price", 
                                f"₹{insights['Price']:,.2f}"
                            )
                            
                            # Calculate from 52-week low
                            pct_from_low = ((insights['Price'] - insights['52 Week Low']) / 
                                           insights['52 Week Low']) * 100
                            
                            metrics_col2.metric(
                                "52-Week Low", 
                                f"₹{insights['52 Week Low']:,.2f}",
                                f"{pct_from_low:+.1f}%"
                            )
                            
                            # Calculate from 52-week high
                            pct_from_high = ((insights['Price'] - insights['52 Week High']) / 
                                            insights['52 Week High']) * 100
                            
                            metrics_col3.metric(
                                "52-Week High", 
                                f"₹{insights['52 Week High']:,.2f}",
                                f"{pct_from_high:+.1f}%"
                            )
                        
                        # Valuation metrics
                        st.subheader("Valuation Metrics")
                        val_col1, val_col2 = st.columns(2)
                        
                        val_col1.metric("P/E Ratio", f"{insights['P/E Ratio']:,.2f}")
                        val_col2.metric("Dividend Yield", 
                                      f"{insights['Dividend Yield']}%" if insights['Dividend Yield'] != 'N/A' 
                                      else "N/A")
                        
                        # Add a button to add this stock to portfolio
                        if st.button("Add to Portfolio"):
                            st.session_state.add_investment_stock = selected_stock
                            st.switch_page("Add Investment")
                    else:
                        st.error(analysis)
            else:
                st.error("Stock data not available")
        
        # Tab 4: Investment Recommendations
        with tab4:
            st.subheader("Investment Recommendations")
            
            # Create two sections
            recom_tab1, recom_tab2 = st.tabs(["Suggested Stocks", "AI Insights"])
            
            with recom_tab1:
                st.write("These stocks have strong fundamentals (Low P/E, High EPS):")
                
                recommended_stocks = self.suggest_investment_opportunities()
                if not recommended_stocks.empty:
                    # Format dataframe for display
                    display_df = recommended_stocks.copy()
                    display_df["Price"] = display_df["Price"].apply(lambda x: f"₹{float(x):,.2f}")
                    display_df["Earnings/Share"] = display_df["Earnings/Share"].apply(lambda x: f"₹{float(x):,.2f}")
                    display_df["Price/Earnings"] = display_df["Price/Earnings"].apply(lambda x: f"{float(x):,.2f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No stock recommendations available")
            
            with recom_tab2:
                # AI-powered recommendations based on portfolio
                user_investments = self.get_investment_data(username)
                
                if not user_investments.empty:
                    portfolio_data = self.calculate_returns(user_investments)
                    
                    # Get AI recommendations
                    with st.spinner("Generating AI insights..."):
                        ai_insights = self.ai_recommendations(portfolio_data)
                        
                    st.markdown("### 🔮 AI-Powered Investment Insights")
                    st.markdown(ai_insights)
                else:
                    st.info("Add investments to your portfolio to receive personalized AI recommendations.")
#module 6: Gamified financial education
import random
import streamlit as st
import webbrowser
from googleapiclient.discovery import build

class Module6LearnTest:
    """Interactive Learning Module for Financial Education"""
    YOUTUBE_API_KEY = "AIzaSyAU5trJXY3u7LoKmKW29oDepZoKn_nrJn4"
    TOPICS = ["Financial Planning", "Investment Knowledge", "Loans for Beginners"]

    QUIZ_QUESTIONS = {
        "Financial Planning": [{"question": "What is the 50-30-20 rule?", 
             "options": ["50% savings, 30% needs, 20% wants", "50% needs, 30% wants, 20% savings", "50% investments, 30% expenses, 20% donations"], 
             "answer": "50% needs, 30% wants, 20% savings"},

            {"question": "Why is an emergency fund important?", 
             "options": ["To buy luxury items", "To cover unexpected expenses", "To increase credit score"], 
             "answer": "To cover unexpected expenses"},

            {"question": "What is a budget?", 
             "options": ["A financial plan", "A type of loan", "An investment strategy"], 
             "answer": "A financial plan"},

            {"question": "Which expense is a 'Need'?", 
             "options": ["Netflix subscription", "Rent payment", "Vacation trip"], 
             "answer": "Rent payment"},

            {"question": "What is a good savings habit?", 
             "options": ["Spend first, save later", "Save a fixed portion of income", "Only save when extra money is left"], 
             "answer": "Save a fixed portion of income"}],  # Same question dictionary as before
        "Investment Knowledge": [{"question": "What is the main goal of investing?", 
             "options": ["To lose money", "To grow wealth", "To avoid taxes"], 
             "answer": "To grow wealth"},

            {"question": "Which is a low-risk investment?", 
             "options": ["Stocks", "Bonds", "Cryptocurrency"], 
             "answer": "Bonds"},

            {"question": "What does 'diversification' mean in investing?", 
             "options": ["Putting money in one stock", "Investing in different assets", "Keeping all money in cash"], 
             "answer": "Investing in different assets"},

            {"question": "What is a mutual fund?", 
             "options": ["A type of bank account", "A professionally managed investment fund", "A form of loan"], 
             "answer": "A professionally managed investment fund"},

            {"question": "What affects stock prices?", 
             "options": ["Company performance", "Moon phases", "Number of employees"], 
             "answer": "Company performance"}],
        "Loans for Beginners": [{"question": "What does APR stand for?", 
             "options": ["Annual Percentage Rate", "Applied Payment Ratio", "Average Profit Return"], 
             "answer": "Annual Percentage Rate"},

            {"question": "Which loan usually has the lowest interest rate?", 
             "options": ["Credit card loan", "Personal loan", "Home loan"], 
             "answer": "Home loan"},

            {"question": "What affects your loan eligibility?", 
             "options": ["Favorite color", "Credit score", "Height"], 
             "answer": "Credit score"},

            {"question": "Which type of loan is secured by property?", 
             "options": ["Student loan", "Auto loan", "Personal loan"], 
             "answer": "Auto loan"},

            {"question": "What happens if you miss a loan payment?", 
             "options": ["Nothing", "Credit score decreases", "You get free money"], 
             "answer": "Credit score decreases"}]
    }

    def __init__(self):
        """Initialize YouTube API"""
        self.youtube = build("youtube", "v3", developerKey=self.YOUTUBE_API_KEY)

    def get_youtube_videos(self, query, max_results=3):
        """Fetch YouTube videos for the selected topic"""
        request = self.youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results)
        response = request.execute()
        videos = [
            {"title": item["snippet"]["title"], "video_id": item["id"]["videoId"]}
            for item in response.get("items", [])
        ]
        return videos

    def start_learning(self):
        """Main function to run the learning module"""
        st.subheader("Select a topic to learn about:")
        selected_topic = st.radio("", self.TOPICS)

        if selected_topic:
            st.success(f"You selected: {selected_topic}")
            st.subheader("🎬 Top 3 YouTube Videos")
            videos = self.get_youtube_videos(selected_topic)

            for idx, video in enumerate(videos):
                st.write(f"{idx + 1}. {video['title']}")
                st.video(f"https://www.youtube.com/watch?v={video['video_id']}")

            if st.button("Start Quiz for " + selected_topic):
                self.run_quiz(selected_topic)

    def run_quiz(self, topic):
        """Runs a 5-question quiz on the selected topic"""
        st.subheader("📝 Quiz Time!")
        quiz_questions = self.QUIZ_QUESTIONS[topic]
        score = 0

        for question_data in quiz_questions:
            st.write(question_data["question"])
            answer = st.radio("Select an option:", question_data["options"], key=random.random())

            if st.button("Submit Answer", key=random.random()):
                if answer == question_data["answer"]:
                    st.success("✅ Correct!")
                    score += 1
                else:
                    st.error(f"❌ Incorrect! The correct answer was: {question_data['answer']}")

        st.subheader(f"🏆 Your final score: {score}/5")
    def run_module6_streamlit(self):
    
        # Create section
        st.title("📚 Financial Learning Module")
    
        # Select a topic
        selected_topic = st.radio(
            "Select a topic to learn about:", 
            self.TOPICS, key="topic_select" # Added Key
        )
    
        if selected_topic:
            st.success(f"You selected: {selected_topic}")
        
            # Show YouTube videos
            st.subheader("🎬 Top 3 YouTube Videos")
            try:
                videos = self.get_youtube_videos(selected_topic)
            
                if not videos:
                    st.warning("No videos found. Please try another topic.")
                else:
                    for idx, video in enumerate(videos):
                        st.write(f"{idx + 1}. {video['title']}")
                        st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
            except Exception as e:
                st.error(f"Error loading videos: {str(e)}")
                st.info("You may need to set up a valid YouTube API key.")
        
            # Quiz section
            st.subheader("📝 Quiz Time!")
            quiz_questions = self.QUIZ_QUESTIONS[selected_topic]
        
            # Initialize score in session state if not present
            if f"quiz_score_{selected_topic}" not in st.session_state:
                st.session_state[f"quiz_score_{selected_topic}"] = 0
                st.session_state[f"quiz_answered_{selected_topic}"] = [False] * len(quiz_questions)
        
            score = st.session_state[f"quiz_score_{selected_topic}"]
            answered = st.session_state[f"quiz_answered_{selected_topic}"]
        
            # Display each question
            for i, question_data in enumerate(quiz_questions):
                question_key = f"question_{i}"
                if i > 0 and not answered[i-1]:
                    continue  # Only show next question after previous is answered
                
                st.write(f"**Question {i+1}:** {question_data['question']}")
            
                # Create a unique key for each radio button
                answer_key = f"{selected_topic}_q{i}_answer"

                default_index = 0  # Set a default index to the first option

                if answer_key not in st.session_state:
                  st.session_state[answer_key] = question_data["options"][default_index] #Initialize
              
                answer = st.radio(
                    "Select an option:", 
                    question_data["options"],
                    key=answer_key,
                )
            
                # Create a unique key for each button
                submit_key = f"{selected_topic}_q{i}_submit"
                if not answered[i] and st.button("Submit Answer", key=submit_key):
                    if answer == question_data["answer"]:
                        st.success("✅ Correct!")
                        st.session_state[f"quiz_score_{selected_topic}"] += 1
                        score = st.session_state[f"quiz_score_{selected_topic}"]
                    else:
                        st.error(f"❌ Incorrect! The correct answer was: {question_data['answer']}")
                
                    # Mark this question as answered
                    answered[i] = True
                    st.session_state[f"quiz_answered_{selected_topic}"] = answered
                    st.rerun()
        
            # Show final score if all questions are answered
            if all(answered):
                st.subheader(f"🏆 Your final score: {score}/{len(quiz_questions)}")
                if score == len(quiz_questions):
                    st.balloons()
                    st.success("Perfect score! You're a financial expert! 🌟")
                elif score >= len(quiz_questions) * 0.7:
                    st.success("Great job! You have a good understanding of this topic. 👍")
                else:
                    st.info("Keep learning! Review the videos and try again. 📚")
                
                if st.button("Reset Quiz", key=f"reset_{selected_topic}"):
                    st.session_state[f"quiz_score_{selected_topic}"] = 0
                    st.session_state[f"quiz_answered_{selected_topic}"] = [False] * len(quiz_questions)
                    for j in range(len(quiz_questions)):
                        if f"{selected_topic}_q{j}_answer" in st.session_state:
                            st.session_state[f"{selected_topic}_q{j}_answer"] = ""
                    st.rerun()