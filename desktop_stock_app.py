import tkinter as tk
from tkinter import scrolledtext, messagebox
import os
import numpy as np # Needed for np.mean if sentiments are combined

# Conditional imports for external APIs for desktop_stock_app.py itself
try:
    from newsapi import NewsApiClient
    from newsapi.newsapi_exception import NewsAPIException
except ImportError:
    NewsApiClient = None
    NewsAPIException = type('NewsAPIException', (Exception,), {}) # Mocking for graceful handling

# --- IMPORTANT: Ensure your 'improved_stock_analysis_script.py' is in the same directory. ---
# We will import functions directly from it.
# If your main script is named 'stock.py', change 'improved_stock_analysis_script' below to 'stock'.
try:
    from improved_stock_analysis_script import (
        get_stock_data, calculate_technical_indicators,
        # fetch_news_sentiment_from_newsapi, # We will use our own wrapper in this GUI app
        fetch_news_sentiment_from_rss,
        analyze_sentiment, analyze_stock, enhanced_analysis, # OpenAI related
        extract_financial_events, assess_impact, generate_signal,
        BASE_GOOGLE_NEWS_RSS_URL,
        NEWS_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY # Import the variables directly
    )
    # Import summarization functions
    from openai_utils import summarize_news_with_deepseek
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not find 'improved_stock_analysis_script.py' or one of its dependencies: {e}. "
                                         "Please ensure it's in the same directory and named correctly.")
    exit() # Exit if the core script isn't found

# Initialize NewsAPI client within the GUI context, if the key is available and NewsApiClient is imported
newsapi_client_gui = None
if NEWS_API_KEY and NewsApiClient: # Check if both key and NewsApiClient class are available
    try:
        newsapi_client_gui = NewsApiClient(api_key=NEWS_API_KEY)
        print("NewsAPI client initialized for GUI app.")
    except Exception as e:
        print(f"Error initializing NewsAPI client for GUI app: {e}")
        messagebox.showwarning("API Init Warning", f"NewsAPI client could not be initialized: {e}. News fetching will be skipped.")
else:
    print("NEWS_API_KEY environment variable not set or 'newsapi-python' library not found for GUI app. NewsAPI functionalities will be skipped.")


class StockAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Desktop Stock Analyzer")
        master.geometry("800x600") # Set initial window size

        # Configure the main window for responsiveness
        master.grid_rowconfigure(2, weight=1)
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1) # Allow ticker entry to expand

        # 1. Ticker Input
        self.ticker_label = tk.Label(master, text="Enter Stock Ticker:", font=("Arial", 12))
        self.ticker_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.ticker_entry = tk.Entry(master, width=25, font=("Arial", 12))
        self.ticker_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.ticker_entry.bind("<Return>", self.analyze_stock_event) # Allows pressing Enter key

        # 2. Analyze Button
        self.analyze_button = tk.Button(master, text="Analyze Stock", command=self.analyze_stock_action,
                                        font=("Arial", 12, "bold"), bg="#4CAF50", fg="white",
                                        activebackground="#45a049", activeforeground="white")
        self.analyze_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        # 3. Output Display Area
        self.output_label = tk.Label(master, text="Analysis Results:", font=("Arial", 12, "bold"))
        self.output_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.output_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=25,
                                                     font=("Courier New", 10), bg="#f0f0f0", fg="#333333",
                                                     borderwidth=2, relief="groove")
        self.output_text.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    def analyze_stock_event(self, event=None): # Added event parameter for bind
        self.analyze_stock_action()

    def analyze_stock_action(self):
        ticker_symbol = self.ticker_entry.get().strip().upper()
        if not ticker_symbol:
            messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
            return

        self.output_text.delete(1.0, tk.END) # Clear previous results
        self.output_text.insert(tk.END, f"Analyzing {ticker_symbol}...\n\n")
        self.output_text.update_idletasks() # Force update to show message immediately

        # --- Call your existing analysis functions from improved_stock_analysis_script.py ---

        historical_data, current_price, company_fundamentals, error = get_stock_data(ticker_symbol)

        if error:
            self.output_text.insert(tk.END, f"Error: {error}\n")
        elif historical_data is None or historical_data.empty:
            self.output_text.insert(tk.END, f"Could not retrieve sufficient data for {ticker_symbol}.\n")
        else:
            self.output_text.insert(tk.END, f"--- Data for {ticker_symbol} ---\n")
            self.output_text.insert(tk.END, f"Current Price: ${current_price:.2f}\n\n")

            technical_indicators = calculate_technical_indicators(historical_data)
            self.output_text.insert(tk.END, "--- Technical Indicators ---\n")
            if technical_indicators:
                for k, v in technical_indicators.items():
                    if v is not None:
                        self.output_text.insert(tk.END, f"{k}: {v:.2f}\n")
                    else:
                        self.output_text.insert(tk.END, f"{k}: Not enough data\n")
            else:
                self.output_text.insert(tk.END, "Technical indicators skipped (TA-Lib not available or insufficient data).\n")
            self.output_text.insert(tk.END, "\n")

            # Call the GUI-specific news fetch wrapper
            newsapi_sentiment, newsapi_titles = self._fetch_news_from_newsapi_for_gui(ticker_symbol)

            ticker_specific_rss_url = BASE_GOOGLE_NEWS_RSS_URL.format(ticker=ticker_symbol)
            rss_sentiment, rss_titles = fetch_news_sentiment_from_rss(ticker_specific_rss_url, ticker_symbol)

            combined_sentiments = []
            combined_news_titles = []
            if newsapi_sentiment is not None:
                combined_sentiments.append(newsapi_sentiment)
                combined_news_titles.extend(newsapi_titles)
            if rss_sentiment is not None:
                combined_sentiments.append(rss_sentiment)
                combined_news_titles.extend(rss_titles)

            overall_news_sentiment = None
            if combined_sentiments:
                overall_news_sentiment = np.mean(combined_sentiments)

            self.output_text.insert(tk.END, "--- News Sentiment ---\n")
            if overall_news_sentiment is not None:
                self.output_text.insert(tk.END, f"Overall News Sentiment: {overall_news_sentiment:.2f}\n")
                self.output_text.insert(tk.END, "Recent News Titles (sample):\n")
                for i, title in enumerate(list(set(combined_news_titles))[:5]): # Show unique top 5 titles
                    self.output_text.insert(tk.END, f"  - {title}\n")
            else:
                self.output_text.insert(tk.END, "Could not fetch news sentiment from any source.\n")
            self.output_text.insert(tk.END, "\n")

            basic_recommendation, basic_confidence, basic_reason = analyze_stock(historical_data, overall_news_sentiment)
            self.output_text.insert(tk.END, f"--- Basic Analysis for {ticker_symbol} ---\n")
            self.output_text.insert(tk.END, f"Recommendation: {basic_recommendation} (Confidence: {basic_confidence}%)\n")
            self.output_text.insert(tk.END, f"Reason: {basic_reason}\n\n")

            social_media_sentiment_placeholder = 0.1 # Placeholder
            enhanced_recommendation, enhanced_confidence, enhanced_reason, alerts = enhanced_analysis(
                ticker_symbol, historical_data, technical_indicators, company_fundamentals,
                overall_news_sentiment, social_media_sentiment_placeholder, combined_news_titles
            )
            self.output_text.insert(tk.END, f"--- Enhanced Analysis for {ticker_symbol} ---\n")
            self.output_text.insert(tk.END, f"Recommendation: {enhanced_recommendation} (Confidence: {enhanced_confidence}%)\n")
            self.output_text.insert(tk.END, f"Reason: {enhanced_reason}\n")
            if alerts:
                self.output_text.insert(tk.END, "Alerts:\n")
                for alert in alerts:
                    self.output_text.insert(tk.END, f"  - {alert}\n")
            else:
                self.output_text.insert(tk.END, "No specific alerts.\n")
            self.output_text.insert(tk.END, "\n")

            # DeepSeek Summary (if DEEPSEEK_API_KEY is set and function available)
            if DEEPSEEK_API_KEY and summarize_news_with_deepseek and combined_news_titles:
                self.output_text.insert(tk.END, f"--- DeepSeek News Summary ---\n")
                deepseek_summary = summarize_news_with_deepseek(combined_news_titles, ticker_symbol)
                self.output_text.insert(tk.END, f"{deepseek_summary if deepseek_summary else 'Could not generate DeepSeek summary.'}\n\n")

            self.output_text.insert(tk.END, "\n")

            # Financial Event Analysis Example
            self.output_text.insert(tk.END, "--- Financial Event Analysis Example ---\n")
            sample_news_content_event = f"""
            {ticker_symbol} announced strong Q1 earnings, beating analyst expectations on both revenue and profit,
            driven by robust advertising growth. However, the company also hinted at potential restructuring
            in its cloud division and faces an ongoing anti-trust lawsuit from the DOJ.
            Analysts remain bullish, but the legal issue adds uncertainty.
            """
            events = extract_financial_events(sample_news_content_event)
            sentiment_for_event = analyze_sentiment(sample_news_content_event)
            impact, event_alerts = assess_impact(events, sentiment_for_event)
            event_signal = generate_signal(impact)

            self.output_text.insert(tk.END, f"Sample News: '{sample_news_content_event[:100]}...'\n")
            self.output_text.insert(tk.END, f"Identified Events: {events}\n")
            self.output_text.insert(tk.END, f"Sentiment for Event: {sentiment_for_event:.2f}\n")
            self.output_text.insert(tk.END, f"Assessed Impact: {impact}\n")
            self.output_text.insert(tk.END, f"Event-based Signal: {event_signal}\n")
            if event_alerts:
                self.output_text.insert(tk.END, "Event Alerts:\n")
                for alert in event_alerts:
                    self.output_text.insert(tk.END, f"  - {alert}\n")
            self.output_text.insert(tk.END, "\nScript finished.\n")
            self.output_text.see(tk.END) # Scroll to the end of the text

    def _fetch_news_from_newsapi_for_gui(self, ticker_symbol: str) -> tuple:
        """
        Wrapper to call NewsAPI using the GUI's initialized client.
        This ensures the correct client (newsapi_client_gui) is used and
        handles NewsAPI-specific exceptions for GUI feedback.
        """
        if not newsapi_client_gui: # Use the client initialized for the GUI app
            print("NewsAPI client not initialized for GUI. Skipping NewsAPI fetch.")
            return None, []

        all_articles = []
        try:
            from datetime import datetime, timedelta # Ensure these are available locally if needed
            articles_response = newsapi_client_gui.get_everything(q=ticker_symbol,
                                                                  language='en',
                                                                  sort_by='relevancy',
                                                                  from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                                                                  to=datetime.now().strftime('%Y-%m-%d'),
                                                                  page_size=20)

            if articles_response and articles_response['articles']:
                all_articles = articles_response['articles']
                sentiments = [analyze_sentiment(article.get('title', '') + " " + (article.get('description', '') or ""))
                              for article in all_articles if article.get('title') or article.get('description')]
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    news_titles = [article.get('title', 'No Title') for article in all_articles]
                    print(f"Fetched {len(news_titles)} articles from NewsAPI for {ticker_symbol} (GUI).")
                    return avg_sentiment, news_titles
            print(f"No recent news found for {ticker_symbol} from NewsAPI (GUI).")
            return None, []
        except NewsAPIException as e: # Specific NewsAPI error handling
            error_message = f"NewsAPI Error: {e.get_message()}"
            messagebox.showerror("NewsAPI Error", error_message)
            print(f"Error fetching news from NewsAPI (GUI): {error_message}")
            return None, []
        except Exception as e: # Catching generic exception for other unexpected errors
            error_message = f"An unexpected error occurred during NewsAPI fetch: {e}"
            messagebox.showerror("NewsAPI Error", error_message)
            print(f"Error fetching news from NewsAPI (GUI): {error_message}")
            return None, []


# --- Main App Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalyzerApp(root)
    root.mainloop() # Starts the Tkinter event loop
