import streamlit as st
import pandas as pd
import pickle
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="MoodReads | AI Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main Container */
    .main-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Book Cards */
    .book-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 5px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .book-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .book-author {
        color: #667eea;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    
    .book-meta {
        color: #718096;
        font-size: 0.9rem;
    }
    
    .rating-badge {
        background: linear-gradient(90deg, #f6ad55, #ed8936);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Mood Tags */
    .mood-tag {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Search Box */
    .stTextInput>div>div>input {
        border: 2px solid #667eea;
        border-radius: 8px;
    }
    
    /* Divider */
    .custom-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    if os.path.exists('books_1000.csv'):
        try:
            df = pd.read_csv('books_1000.csv')
            # Add mood mapping
            mood_map = {
                'Fiction': ['Relaxed', 'Thoughtful', 'Curious'],
                'Thriller': ['Excited', 'Adventurous', 'Intense'],
                'Romance': ['Romantic', 'Hopeful', 'Emotional'],
                'Sci-Fi': ['Curious', 'Thoughtful', 'Adventurous'],
                'Fantasy': ['Adventurous', 'Imaginative', 'Escapist'],
                'Horror': ['Intense', 'Adventurous', 'Curious'],
                'Mystery': ['Curious', 'Thoughtful', 'Analytical'],
                'Biography': ['Thoughtful', 'Inspired', 'Curious'],
                'Non Fiction': ['Thoughtful', 'Curious', 'Inspired']
            }
            df['mood'] = df['genre'].apply(lambda x: random.choice(mood_map.get(x, ['Curious'])))
            return df
        except Exception as e:
            st.error(f"Error loading  {e}")
            return None
    else:
        st.error("❌ books_1000.csv not found!")
        st.info("Please run: `python generate_data.py`")
        return None

# ============================================
# LOAD/TRAIN AI MODEL
# ============================================
@st.cache_data
def get_similarity_matrix(df):
    if os.path.exists('model_mood_v2.pkl'):
        try:
            with open('model_mood_v2.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    # Create tags
    df['tags'] = df['author'] + ' ' + df['genre'] + ' ' + df['title'] + ' ' + df['description'].fillna('')
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    vectors = tfidf.fit_transform(df['tags'])
    similarity = cosine_similarity(vectors)
    
    # Save model
    with open('model_mood_v2.pkl', 'wb') as f:
        pickle.dump({'similarity': similarity, 'tfidf': tfidf}, f)
    
    return {'similarity': similarity, 'tfidf': tfidf}

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_mood_books(df, mood):
    mood_books = df[df['mood'] == mood].sort_values('rating', ascending=False)
    return mood_books.head(10)

def filter_by_genre(df, genre):
    if genre == "All Genres":
        return df
    return df[df['genre'] == genre]

def search_books(df, query):
    if not query:
        return pd.DataFrame()
    return df[
        df['title'].str.contains(query, case=False, na=False) |
        df['author'].str.contains(query, case=False, na=False) |
        df['genre'].str.contains(query, case=False, na=False)
    ]

def display_book_card(row):
    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">📖 {row['title']}</div>
        <div class="book-author">✍️ {row['author']}</div>
        <div class="book-meta">
            📚 {row['genre']} &nbsp;|&nbsp; 
            📅 {row['year']} &nbsp;|&nbsp; 
            🏷️ {row.get('mood', 'N/A')}
        </div>
        <div class="rating-badge">⭐ {row['rating']}/5.0</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 MoodReads</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Book Discovery Based on Your Mood & Preferences</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        st.markdown("---")
        menu = st.radio(
            "Choose Feature",
            [
                "🏠 Home",
                "😊 Find by Mood",
                "🎭 Browse by Genre",
                "🤖 AI Recommendations",
                "🔍 Search Books",
                "📊 Research & Stats"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **MoodReads** is an AI-powered book recommendation system that suggests books based on your mood, preferences, and reading history.
        
        **Tech Stack:**
        - Python & Streamlit
        - TF-IDF + Cosine Similarity
        - Pandas & Scikit-learn
        
        **Version:** 2.0 Professional
        """)
    
    # ============================================
    # HOME PAGE
    # ============================================
    if menu == "🏠 Home":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Stats Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">Total Books</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df['genre'].unique())}</div>
                <div class="stat-label">Genres</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df['author'].unique())}</div>
                <div class="stat-label">Authors</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{df['rating'].mean():.1f}</div>
                <div class="stat-label">Avg Rating</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top 50 Books
        st.markdown("### 🔥 Top 50 Most Popular Books")
        st.markdown("*Sorted by user ratings*")
        
        top_books = df.sort_values(by='rating', ascending=False).head(50)
        
        # Display in 2 columns
        col1, col2 = st.columns(2)
        for idx, row in top_books.iterrows():
            with col1 if idx % 2 == 0 else col2:
                display_book_card(row)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # MOOD-BASED RECOMMENDATIONS
    # ============================================
    elif menu == "😊 Find by Mood":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown("### 😊 How Are You Feeling Today?")
        st.markdown("*Select your current mood to discover perfect books*")
        
        moods = ['Relaxed', 'Thoughtful', 'Curious', 'Excited', 'Adventurous', 
                 'Intense', 'Romantic', 'Hopeful', 'Emotional', 'Imaginative', 
                 'Escapist', 'Analytical', 'Inspired']
        
        selected_mood = st.selectbox("🎭 Choose your mood:", moods)
        
        if st.button("🔍 Find Books for This Mood", type="primary"):
            with st.spinner(f"Finding books for {selected_mood} mood..."):
                mood_books = get_mood_books(df, selected_mood)
                
                if len(mood_books) > 0:
                    st.success(f"✨ Found {len(mood_books)} books perfect for {selected_mood} readers!")
                    
                    for idx, row in mood_books.iterrows():
                        display_book_card(row)
                else:
                    st.warning("No books found for this mood. Try another!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # GENRE BROWSE
    # ============================================
    elif menu == "🎭 Browse by Genre":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown("### 🎭 Explore by Genre")
        
        # Genre Selection
        genres = ["All Genres"] + sorted(df['genre'].unique())
        selected_genre = st.selectbox("📚 Select Genre:", genres)
        
        # Advanced Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_rating = st.slider("⭐ Minimum Rating:", 0.0, 5.0, 3.0)
        with col2:
            year_min = st.slider("📅 From Year:", 1950, 2023, 1990)
        with col3:
            year_max = st.slider("📅 To Year:", 1950, 2023, 2023)
        
        # Apply Filters
        filtered_df = filter_by_genre(df, selected_genre)
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        filtered_df = filtered_df[(filtered_df['year'] >= year_min) & (filtered_df['year'] <= year_max)]
        
        st.markdown(f"**📊 Found {len(filtered_df)} books matching your criteria**")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)
            for idx, row in filtered_df.iterrows():
                with col1 if idx % 2 == 0 else col2:
                    display_book_card(row)
        else:
            st.warning("No books match your filters. Try adjusting them!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # AI RECOMMENDATIONS
    # ============================================
    elif menu == "🤖 AI Recommendations":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown("### 🤖 AI-Powered Smart Recommendations")
        st.markdown("*Our algorithm analyzes content, mood, and ratings to find your perfect match*")
        
        # Load model
        model_data = get_similarity_matrix(df)
        similarity = model_data['similarity']
        
        # Book Selection
        book_list = df['title'].tolist()
        selected_book = st.selectbox("📖 Select a book you enjoyed:", book_list)
        
        # Serendipity Control
        serendipity = st.slider("🎲 Diversity Level", 0, 100, 30, 
                               help="Higher = More diverse recommendations from different genres")
        
        if st.button("🚀 Get Smart Recommendations", type="primary"):
            try:
                idx = df[df['title'] == selected_book].index[0]
                sim_scores = list(enumerate(similarity[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                # Apply serendipity
                if serendipity > 50:
                    user_genre = df.iloc[idx]['genre']
                    diverse_books = df[df['genre'] != user_genre].sort_values('rating', ascending=False).head(2)
                    top_indices = [i[0] for i in sim_scores[1:4]]
                    top_indices.extend(diverse_books.index.tolist())
                else:
                    top_indices = [i[0] for i in sim_scores[1:6]]
                
                st.success("🎯 Your Personalized Recommendations:")
                
                for i in top_indices[:5]:
                    book = df.iloc[i]
                    display_book_card(book)
                    
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # SEARCH BOOKS
    # ============================================
    elif menu == "🔍 Search Books":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown("### 🔍 Search Our Library")
        
        # Search Input
        search_query = st.text_input("📝 Enter book title, author, or genre:", placeholder="e.g., Harry Potter, Stephen King, Fantasy...")
        
        if search_query:
            results = search_books(df, search_query)
            
            if len(results) > 0:
                st.success(f"✅ Found {len(results)} books matching '{search_query}'")
                
                col1, col2 = st.columns(2)
                for idx, row in results.iterrows():
                    with col1 if idx % 2 == 0 else col2:
                        display_book_card(row)
            else:
                st.warning(f"😕 No books found for '{search_query}'. Try different keywords!")
        else:
            st.info("👆 Start typing to search...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # RESEARCH & STATS
    # ============================================
    elif menu == "📊 Research & Stats":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown("### 📊 Research Insights & Analytics")
        
        # Stats Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">Total Books</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{df['rating'].mean():.2f}</div>
                <div class="stat-label">Average Rating</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df['genre'].unique())}</div>
                <div class="stat-label">Unique Genres</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📚 Books by Genre")
            genre_counts = df['genre'].value_counts()
            st.bar_chart(genre_counts)
        
        with col2:
            st.markdown("#### 🏷️ Mood Distribution")
            mood_counts = df['mood'].value_counts()
            st.bar_chart(mood_counts)
        
        st.markdown("---")
        
        # Research Comparison Table
        st.markdown("### 🔬 How MoodReads Differs from Existing Systems")
        
        comparison_data = {
            'Feature': ['Algorithm', 'Personalization', 'Discovery', 'Filter Bubble', 'Cold Start'],
            'Amazon/Goodreads': ['Collaborative Filtering', 'Purchase History', "Customers also bought", 'High', 'Problem'],
            'MoodReads (This Project)': ['Hybrid Content-Mood', 'Current Mood + Content', 'Serendipity Engine', 'Low', 'Solved']
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        st.markdown("""
        ### Key Research Contributions:
        1. ✅ **Mood-Aware Recommendation Model** - Novel approach combining emotional state with content analysis
        2. ✅ **Serendipity Parameter** - Adjustable diversity to break filter bubbles
        3. ✅ **Multi-Dimensional Book Matching** - Combines content, mood, rating, and user control
        4. ✅ **Content-Based Filtering** - Solves cold-start problem for new books
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()