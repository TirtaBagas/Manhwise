import streamlit as st
import streamlit.components.v1 as components
import joblib as j
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the files
main_df = j.load('main_df.jbl')
personal_df = j.load('personal_df.jbl')
matched_value = j.load('matched_value.pkl')

# Genre list
genre_list = ['Action', 'Adventure', 'Comedy', 'Drama', 'Ecchi', 'Fantasy', 'Horror',
       'Mahou Shoujo', 'Mecha', 'Music', 'Mystery', 'Psychological', 'Romance',
       'Sci-Fi', 'Slice Of Life', 'Sports', 'Supernatural', 'Thriller']

# Recommendation by title
def recommend_by_title_cosine(title, main_df, personal_df, matched_value, n_recommendations=10):
    """
    Mendapatkan rekomendasi berdasarkan judul
    """
    try:
        # Cari index dari judul di main_df
        idx = main_df[main_df['Title'].str.lower() == title.lower()].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(matched_value[idx]))
        sim_scores_filtered = [score for score in sim_scores if score[0] != idx]
        sim_scores_sorted = sorted(sim_scores_filtered, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_scores = sim_scores_sorted[:n_recommendations]
        movie_indices = [i[0] for i in top_scores]
        
        
        recommendations = personal_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [score[1] for score in top_scores]

        recommendations = recommendations.sort_values(by=['similarity_score', 'Popularity'], ascending=[False, False]).head(n_recommendations)
        
        return recommendations, None
        
    except IndexError:
        return None, f"'{title}' tidak ditemukan dalam database"
    except Exception as e:
        return None, f"Terjadi error saat mencari rekomendasi: {e}"

# Recommendation by genres
def recommend_by_genres(selected_genres, top_n=10):
    if not selected_genres:
        return []

    # Format genre dalam main_df
    main_df['Genre List'] = main_df['Genres'].fillna('').apply(lambda x: [g.strip() for g in x.split(',')])

    # Hitung banyak genre yang cocok
    def count_matches(genres):
        return len(set(genres).intersection(set(selected_genres)))

    personal_df['Matching Genres Count'] = main_df['Genre List'].apply(count_matches)

    # Filter yang memiliki minimal 1 genre cocok
    filtered = personal_df[personal_df['Matching Genres Count'] > 0].copy()

    if filtered.empty:
        return []

    # Urutkan hasil
    filtered = filtered.sort_values(by=['Matching Genres Count', 'Popularity'], ascending=[False, False]).head(top_n)

    return filtered

# Format number
def format_number(num):
    return f"{int(num):,}" if num >= 1000 else str(int(num))

# Styles
st.markdown(
    """
    <style>
    /* Background image untuk seluruh halaman */
    .stApp {
        background-image: url("https://images.saymedia-content.com/.image/t_share/MjAwMzExMTA0OTY1MTI1NDk2/the-best-manhwa-with-op-mc-you-must-read.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    .block-container {
        max-width: 900px;
        margin: 10rem auto;
        padding: 2rem 3rem;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.2);
    }
    .info-text {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 50px;
    }
    .title-text {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
        min-height: 50px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        position: relative;
        display: block;
    }
    .title-text:hover::after {
        content: attr(data-fulltitle);
        position: absolute;
        background: #333;
        color: #fff;
        padding: 5px 10px;
        border-radius: 5px;
        top: -35px;
        left: 50%;
        transform: translateX(-50%);
        white-space: normal;
        max-width: 300px;
        text-align: center;
        z-index: 10;
        font-weight: normal;
    }
    </style>
""", unsafe_allow_html=True)

img_style = """
    display: block;
    margin-left: auto;
    margin-right: auto;
    height: 220px;
    width: 160px;
    object-fit: fill;
    transition: transform 0.3s ease;
"""

# Streamlit app
st.title("Manhwise")
st.write("Select a title or choose genres to get recommendations!")

option = st.radio("Choose recommendation type", ('By Title', 'By Genres'))

# Update render_title
def render_title(title):
    if len(title) > 15:
        short_title = title[:30] + "..."
        return st.markdown(f"""
            <div style='text-align: center; font-size: 16px; font-weight: bold; margin-top: 10px; min-height: 50px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                <span title="{title}">{short_title}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        return st.markdown(f"""
            <div style='text-align: center; font-size: 16px; font-weight: bold; margin-top: 10px; min-height: 50px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                {title}
            </div>
        """, unsafe_allow_html=True)

if option == 'By Title':
    title_list = personal_df['Title'].drop_duplicates().tolist()
    selected_title = st.selectbox(
        "Select a Manhwa Title", 
        options=title_list, 
        index=None,
        placeholder="Pilih judul"
    )

    if st.button('Recommend'):
        if selected_title:
            # 1. Panggil fungsi dengan argumen lengkap dan tampung 2 hasilnya
            recs, error = recommend_by_title_cosine(selected_title, main_df, personal_df, matched_value)

            # 2. Cek apakah ada error
            if error:
                st.error(error)
            else:
                st.subheader("Berikut adalah beberapa rekomendasi untuk Anda:")
                n_cols = 5
            for i in range(0, len(recs), n_cols):
                cols = st.columns(n_cols)
                for idx, row in enumerate(recs.iloc[i:i+n_cols].iterrows()):
                    row = row[1]
                    with cols[idx]:
                        render_title(row['Title'])
                        cover_url = row['Cover Image']
                        if isinstance(cover_url, str) and cover_url.startswith('http'):
                            st.markdown(f"<img src='{cover_url}' style='{img_style}' />", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; color: #fff; margin-bottom: 10px;'>[No Image]</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <p class='info-text'>
                            Genre: {row['Genres']}<br>
                            Popularity: {format_number(row['Popularity']) if pd.notnull(row['Popularity']) else '-'}
                        </p>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Silakan pilih judul terlebih dahulu.")
else:
    selected_genres = st.multiselect("Select up to 5 Genres", genre_list)
    if len(selected_genres) > 5:
        st.error("You can select up to 5 genres only!")
    elif st.button('Recommend') and selected_genres:
        recommendations = recommend_by_genres(selected_genres)
        st.subheader("Here are a few recommendations:")
        n_cols = 5
        for i in range(0, len(recommendations), n_cols):
            cols = st.columns(n_cols)
            for idx, row in enumerate(recommendations.iloc[i:i+n_cols].iterrows()):
                row = row[1]
                with cols[idx]:
                    render_title(row['Title'])
                    cover_url = row['Cover Image']
                    if isinstance(cover_url, str) and cover_url.startswith('http'):
                        st.markdown(f"<img src='{cover_url}' style='{img_style}' />", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='text-align: center; color: #fff; margin-bottom: 10px;'>[No Image]</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <p class='info-text'>
                            Genre: {row['Genres']}<br>
                            Popularity: {format_number(row['Popularity']) if pd.notnull(row['Popularity']) else '-'}
                        </p>
                    """, unsafe_allow_html=True)


                
st.markdown("</div>", unsafe_allow_html=True)