import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import time

st.set_page_config(
    page_title="Amazon Kids Product Search",
    page_icon="🎨",
    layout="wide"
)

# ============================================================
# CREDENTIALS FROM STREAMLIT SECRETS
# ============================================================
try:
    CLOUD_ID = st.secrets["CLOUD_ID"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
except KeyError as e:
    st.error(f"⚠️ Missing secret: '{e}'. Please configure CLOUD_ID, USERNAME, and PASSWORD in Streamlit Cloud dashboard.")
    st.stop()

INDEX_NAME = "amazon_2020_bbq"

# ============================================================
# INITIALIZE CONNECTIONS (cached so they only run once)
# ============================================================
@st.cache_resource
def init_elasticsearch():
    return Elasticsearch(
        cloud_id=CLOUD_ID,
        basic_auth=(USERNAME, PASSWORD),
        request_timeout=60
    )

@st.cache_resource
def init_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = init_elasticsearch()
model  = init_model()

# ============================================================
# SEARCH FUNCTIONS
# ============================================================
def bm25_search(query, k=10):
    try:
        response = client.search(
            index=INDEX_NAME,
            body={
                "size": k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["product_name^3", "brand^2", "category^1.5", "document_text"],
                        "type": "best_fields"
                    }
                },
                "_source": ["product_name", "brand", "price", "category", "image_url"]
            }
        )
        return response["hits"]["hits"]
    except Exception as e:
        st.error(f"BM25 search error: {e}")
        return []

def vector_search(query, k=10):
    try:
        query_vector = model.encode(query).tolist()
        response = client.search(
            index=INDEX_NAME,
            body={
                "size": k,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": 100
                },
                "_source": ["product_name", "brand", "price", "category", "image_url"]
            }
        )
        return response["hits"]["hits"]
    except Exception as e:
        st.error(f"Vector search error: {e}")
        return []

def hybrid_rrf_search(query, k=10):
    try:
        query_vector = model.encode(query).tolist()
        response = client.search(
            index=INDEX_NAME,
            body={
                "size": k,
                "retriever": {
                    "rrf": {
                        "retrievers": [
                            {
                                "standard": {
                                    "query": {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["product_name^3", "brand^2", "category^1.5", "document_text"]
                                        }
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "field": "embedding",
                                    "query_vector": query_vector,
                                    "k": 50,
                                    "num_candidates": 100
                                }
                            }
                        ],
                        "rank_window_size": 100
                    }
                },
                "_source": ["product_name", "brand", "price", "category", "image_url"]
            }
        )
        return response["hits"]["hits"]
    except Exception as e:
        st.error(f"Hybrid RRF search error: {e}")
        return []

def full_pipeline_search(query, k=10):
    try:
        query_vector = model.encode(query).tolist()
        response = client.search(
            index=INDEX_NAME,
            body={
                "size": k,
                "retriever": {
                    "text_similarity_reranker": {
                        "retriever": {
                            "rrf": {
                                "retrievers": [
                                    {
                                        "standard": {
                                            "query": {
                                                "multi_match": {
                                                    "query": query,
                                                    "fields": ["product_name^3", "brand^2", "category^1.5", "document_text"]
                                                }
                                            }
                                        }
                                    },
                                    {
                                        "knn": {
                                            "field": "embedding",
                                            "query_vector": query_vector,
                                            "k": 50,
                                            "num_candidates": 100
                                        }
                                    }
                                ],
                                "rank_window_size": 100
                            }
                        },
                        "field": "document_text",
                        "inference_id": "jina_reranker_v3",
                        "inference_text": query,
                        "rank_window_size": 50
                    }
                },
                # document_text MUST be in _source for the reranker to work
                "_source": ["product_name", "brand", "price", "category", "image_url", "document_text"]
            }
        )
        return response["hits"]["hits"]
    except Exception as e:
        st.error(f"Full Pipeline search error: {e}")
        st.warning("💡 Falling back to Hybrid RRF...")
        return hybrid_rrf_search(query, k=k)

# ============================================================
# UI - HEADER
# ============================================================
st.title("🎨 Amazon Kids Product Search")
st.markdown("### Powered by BBQ Quantization + Hybrid RRF + JinaAI Reranker")
st.markdown("---")

# ============================================================
# SEARCH BAR (before sidebar so st.stop() works cleanly)
# ============================================================
query = st.text_input(
    "🔍 Search for kids products:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., LEGO sets, educational toys, board games..."
)

# Stop here if query is empty - prevents BadRequestError
if not query or not query.strip():
    st.info("👆 Enter a search query above to find products!")
    st.markdown("### 💡 Try searching for:")
    cols = st.columns(4)
    suggestions = [
        "Hot Wheels race track", "Barbie dolls",
        "LEGO Star Wars", "educational STEM toys",
        "coloring books", "Pokemon plush",
        "board games family", "outdoor sports toys"
    ]
    for i, sug in enumerate(suggestions):
        with cols[i % 4]:
            st.markdown(f"- {sug}")
    st.stop()

# ============================================================
# SIDEBAR (after query check)
# ============================================================
st.sidebar.header("⚙️ Configuration")
comparison_mode = st.sidebar.checkbox("🔥 Compare All Methods", value=False)

if not comparison_mode:
    search_method = st.sidebar.selectbox(
        "Search Method",
        ["Full Pipeline", "Hybrid RRF", "Vector Semantic", "BM25 Keyword"]
    )
else:
    search_method = None

num_results = st.sidebar.slider("Number of Results", 3, 12, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Quick Searches:**")
examples = [
    "Hot Wheels race track",
    "Barbie dolls",
    "LEGO Star Wars",
    "educational STEM toys",
    "coloring books",
    "Pokemon plush",
    "board games family",
    "outdoor sports toys"
]

for ex in examples:
    if st.sidebar.button(ex, key=f"btn_{ex}"):
        st.session_state["query"] = ex
        st.rerun()

# ============================================================
# COMPARISON MODE
# ============================================================
if comparison_mode:
    st.markdown(f"### 🔬 Comparing all methods for: `{query}`")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**📝 BM25 Keyword**")
        with st.spinner("Searching..."):
            start = time.time()
            bm25_res = bm25_search(query, k=num_results)
            bm25_time = (time.time() - start) * 1000
        st.metric("Latency", f"{bm25_time:.0f}ms")
        st.caption(f"{len(bm25_res)} results")
        for i, hit in enumerate(bm25_res, 1):
            s = hit["_source"]
            name = s.get('product_name', 'N/A')
            st.markdown(f"**{i}.** {name[:45]}{'...' if len(name) > 45 else ''}")
            st.caption(f"${s.get('price', 0):.2f}")

    with col2:
        st.markdown("**🧠 Vector Semantic**")
        with st.spinner("Searching..."):
            start = time.time()
            vec_res = vector_search(query, k=num_results)
            vec_time = (time.time() - start) * 1000
        st.metric("Latency", f"{vec_time:.0f}ms")
        st.caption(f"{len(vec_res)} results")
        for i, hit in enumerate(vec_res, 1):
            s = hit["_source"]
            name = s.get('product_name', 'N/A')
            st.markdown(f"**{i}.** {name[:45]}{'...' if len(name) > 45 else ''}")
            st.caption(f"${s.get('price', 0):.2f}")

    with col3:
        st.markdown("**🔀 Hybrid RRF**")
        with st.spinner("Searching..."):
            start = time.time()
            hyb_res = hybrid_rrf_search(query, k=num_results)
            hyb_time = (time.time() - start) * 1000
        st.metric("Latency", f"{hyb_time:.0f}ms")
        st.caption(f"{len(hyb_res)} results")
        for i, hit in enumerate(hyb_res, 1):
            s = hit["_source"]
            name = s.get('product_name', 'N/A')
            st.markdown(f"**{i}.** {name[:45]}{'...' if len(name) > 45 else ''}")
            st.caption(f"${s.get('price', 0):.2f}")

    with col4:
        st.markdown("**🚀 Full Pipeline**")
        with st.spinner("Reranking with Jina AI..."):
            start = time.time()
            pip_res = full_pipeline_search(query, k=num_results)
            pip_time = (time.time() - start) * 1000
        st.metric("Latency", f"{pip_time:.0f}ms")
        st.caption(f"{len(pip_res)} results")
        for i, hit in enumerate(pip_res, 1):
            s = hit["_source"]
            name = s.get('product_name', 'N/A')
            st.markdown(f"**{i}.** {name[:45]}{'...' if len(name) > 45 else ''}")
            st.caption(f"${s.get('price', 0):.2f}")

    st.markdown("---")
    st.success("💡 Full Pipeline = Hybrid RRF candidates → Jina AI reranked for maximum relevance!")

# ============================================================
# SINGLE METHOD MODE
# ============================================================
elif not comparison_mode and search_method:
    funcs = {
        "Full Pipeline":    full_pipeline_search,
        "Hybrid RRF":       hybrid_rrf_search,
        "Vector Semantic":  vector_search,
        "BM25 Keyword":     bm25_search
    }

    method_icons = {
        "Full Pipeline":    "🚀",
        "Hybrid RRF":       "🔀",
        "Vector Semantic":  "🧠",
        "BM25 Keyword":     "📝"
    }

    st.markdown(f"### {method_icons[search_method]} {search_method} results for: `{query}`")

    with st.spinner(f"Searching with {search_method}..."):
        start   = time.time()
        results = funcs[search_method](query, k=num_results)
        latency = (time.time() - start) * 1000

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ Latency",  f"{latency:.0f}ms")
    m2.metric("📊 Results",  len(results))
    m3.metric("🔧 Method",   search_method)
    m4.metric("🎯 Storage",  "BBQ int8")

    st.markdown("---")

    if not results:
        st.warning("No results found. Try a different search term.")
    else:
        for i, hit in enumerate(results, 1):
            s = hit["_source"]

            col_img, col_info = st.columns([1, 4])

            with col_img:
                img = s.get("image_url", "")
                if img and str(img).startswith("http"):
                    try:
                        st.image(img, width=120)
                    except:
                        st.write("🖼️")
                else:
                    st.write("🖼️")

            with col_info:
                product_name = s.get('product_name', 'Unknown Product')
                st.markdown(f"### {i}. {product_name}")

                d1, d2, d3, d4 = st.columns(4)

                brand = s.get('brand', 'N/A')
                brand = 'N/A' if str(brand) in ['nan', 'None', ''] else brand

                category = s.get('category', 'N/A')
                category = 'N/A' if str(category) in ['nan', 'None', ''] else str(category)[:40]

                price = s.get('price', 0)
                try:
                    price = float(price)
                except:
                    price = 0.0

                score = hit.get('_score', 0)
                try:
                    score = float(score) if score is not None else 0.0
                except:
                    score = 0.0

                d1.markdown(f"**Brand:** {brand}")
                d2.markdown(f"**Price:** ${price:.2f}")
                d3.markdown(f"**Category:** {category}")
                d4.markdown(f"**Score:** {score:.4f}")

            st.markdown("---")

# ============================================================
# SIDEBAR FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Architecture")
st.sidebar.markdown("""
**3-Stage Pipeline:**
1. 🔍 BM25 Keyword Search
2. 🧠 Vector Semantic Search  
3. 🔀 Hybrid RRF Fusion
4. 🚀 Jina AI Reranker v3

**Features:**
- BBQ int8_hnsw (75% storage savings)
- Semantic + keyword fusion
- Cross-encoder reranking via Jina AI
- Elasticsearch native inference
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**🏆 Elastic Blogathon 2026**")
