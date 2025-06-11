import streamlit as st
import google.generativeai as genai
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import re

# Configure page
st.set_page_config(
    page_title="Query Fan-Out Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for caching
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

def safe_rerun():
    """Handle different Streamlit versions for rerun."""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

@dataclass
class GenerationMetadata:
    total_queries_generated: int
    generation_time_ms: int
    model_used: str
    prompt_version: str

@dataclass
class SyntheticQuery:
    query: str
    type: str
    user_intent: str
    reasoning: str
    confidence_score: float

class QueryFanOutGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use relaxed safety settings for query generation
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        self.model = genai.GenerativeModel('gemini-2.0-flash', safety_settings=safety_settings)
        self.prompt_version = "1.0.0"
        
    def _build_cot_prompt(
        self,
        primary_keyword: str,
        search_mode: str,
        industry_context: Optional[str] = None,
        query_intent: Optional[str] = None,
        user_persona: Optional[str] = None
    ) -> str:
        """
        Build a single zero-shot Chain-of-Thought prompt that:
        1) Repeats the primary query five times for bias
        2) Asks for step-by-step reasoning
        3) Instructs to output exactly 10-15 (AI_Overview) or 15-20 (AI_Mode) queries in JSON
        """
        # 1) Repeat the query five times for bias
        repeated = " ".join([primary_keyword] * 5)

        lines = [
            f'Repeated Query Bias: "{repeated}"'
        ]

        # 2) Base instruction with step-by-step reasoning
        lines.append(f'Answer the following query: "{primary_keyword}"')
        lines.append(
            "Give your reasoning step by step—explain how each distinct angle, "
            "facet, or subtopic is discovered."
        )

        # 3) Add optional industry/query_intent/persona context
        if industry_context:
            lines.append(f"Industry Context: {industry_context}")
        if query_intent:
            guidance = {
                "informational": "Focus on definitions, background, and explanatory facets.",
                "commercial": "Emphasize price, comparison, and product/vendor alternatives.",
                "transactional": "Generate queries about where/how to buy, availability, pricing.",
                "navigational": "Create brand/site-specific or service-specific expansions."
            }.get(query_intent, "")
            lines.append(f"Query Intent: {query_intent} — {guidance}")
        if user_persona:
            lines.append(f"User Persona: {user_persona}")

        # 4) Specify target number of queries based on search_mode
        if search_mode == "AI_Mode":
            lines.append("Generate 15–20 complex and diverse queries that cover all relevant facets.")
        else:  # AI_Overview
            lines.append("Generate 10–15 focused queries to provide a comprehensive AI Overview.")

        # 5) Instruct output format in JSON
        lines.append(
            "Finally, output valid JSON in exactly this format:\n"
            '''{
  "query_count_reasoning": "Your explanation of why you chose exactly N queries",
  "synthetic_queries": [
    {
      "query": "…",
      "type": "one of: reformulation|related_query|implicit_query|comparative_query|entity_expansion|personalized_query",
      "user_intent": "…",
      "reasoning": "…",
      "confidence_score": 0.0–1.0
    },
    …
  ]
}'''
        )

        return "\n\n".join(lines)
    
    def _validate_query(self, query: str) -> bool:
        """Validate a single query."""
        words = query.strip().split()
        if len(words) < 2 or len(words) > 15:
            return False
        if len(query) < 5:
            return False
        # Must contain alphabetic characters
        if not re.search(r'[a-zA-Z]', query):
            return False
        return True
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate simple word-based Jaccard similarity between two queries."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)
    
    def _filter_queries(self, primary_keyword: str, queries: List[SyntheticQuery]) -> List[SyntheticQuery]:
        """Apply quality filters: validate, remove identical, dedupe by similarity."""
        filtered = []
        primary_words = set(primary_keyword.lower().split())
        
        for query in queries:
            # Validate structure
            if not self._validate_query(query.query):
                continue
                
            # Skip if identical to primary keyword
            query_words = set(query.query.lower().split())
            if query_words == primary_words:
                continue
                
            # Check for duplicates against already filtered
            is_duplicate = False
            for existing in filtered:
                if self._calculate_similarity(query.query, existing.query) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(query)
                
        return filtered
    
    def _generate_cache_key(self, primary_keyword: str, **params) -> str:
        """Generate a deterministic cache key from parameters."""
        cache_data = {
            'primary_keyword': primary_keyword.lower(),
            **params
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def generate_fanout(
        self,
        primary_keyword: str,
        search_mode: str = "AI_Overview",
        industry_context: Optional[str] = None,
        query_intent: Optional[str] = None,
        user_persona: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Generate synthetic queries for a primary keyword."""
        # 1) Compute cache key
        cache_key = self._generate_cache_key(
            primary_keyword,
            search_mode=search_mode,
            industry_context=industry_context,
            query_intent=query_intent,
            user_persona=user_persona
        )
        
        if use_cache and cache_key in st.session_state.cache:
            cached_data = st.session_state.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(hours=24):
                return cached_data['data']
        
        # 2) Build the CoT prompt
        prompt = self._build_cot_prompt(
            primary_keyword,
            search_mode,
            industry_context,
            query_intent,
            user_persona
        )
        
        # 3) Generate with retry logic
        start_time = time.time()
        max_retries = 3
        retry_delay = 2
        
        try:
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    generation_time_ms = int((time.time() - start_time) * 1000)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if ("429" in msg or "quota" in msg or "rate" in msg) and attempt < max_retries - 1:
                        st.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise e
            
            # 4) Parse response text for JSON
            response_text = response.text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1]
                if '```' in response_text:
                    response_text = response_text.split('```')[0]
            
            json_match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in model response")
            
            # 5) Convert each synthetic query dict into a dataclass
            synthetic_queries: List[SyntheticQuery] = []
            for sq in response_data.get('synthetic_queries', []):
                synthetic_queries.append(SyntheticQuery(
                    query=sq['query'],
                    type=sq['type'],
                    user_intent=sq.get('user_intent', ""),
                    reasoning=sq['reasoning'],
                    confidence_score=float(sq.get('confidence_score', 0.8))
                ))
            
            # 6) Apply filters (validate + dedupe)
            filtered_queries = self._filter_queries(primary_keyword, synthetic_queries)
            
            # 7) Build final result structure
            result = {
                'primary_keyword': primary_keyword,
                'query_count_reasoning': response_data.get('query_count_reasoning', ""),
                'synthetic_queries': [asdict(q) for q in filtered_queries],
                'generation_metadata': asdict(GenerationMetadata(
                    total_queries_generated=len(filtered_queries),
                    generation_time_ms=generation_time_ms,
                    model_used='gemini-1.5-flash',
                    prompt_version=self.prompt_version
                ))
            }
            
            # 8) Cache the result
            if use_cache:
                st.session_state.cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
            
            # 9) Append to history
            st.session_state.generation_history.append({
                'timestamp': datetime.now(),
                'primary_keyword': primary_keyword,
                'queries_generated': len(filtered_queries),
                'search_mode': search_mode
            })
            
            return result
        
        except Exception as e:
            st.error(f"Error generating queries: {str(e)}")
            return None
    
    def process_batch(
        self,
        keywords: List[str],
        search_mode: str = "AI_Overview",
        batch_size: int = 10,
        industry_context: Optional[str] = None,
        query_intent: Optional[str] = None,
        user_persona: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """Process multiple keywords in batches."""
        results = []
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            batch_results = []
            
            for keyword in batch:
                result = self.generate_fanout(
                    primary_keyword=keyword,
                    search_mode=search_mode,
                    industry_context=industry_context,
                    query_intent=query_intent,
                    user_persona=user_persona,
                    use_cache=use_cache
                )
                if result:
                    batch_results.append(result)
                time.sleep(1)  # Rate limiting
                
            results.extend(batch_results)
            
        return results

# Streamlit UI
def main():
    st.title("🔍 Query Fan-Out Generator")
    st.markdown("""
    Generate synthetic queries from primary keywords using Chain-of-Thought prompting,
    mimicking Google's AI Overview algorithm.
    
    **Target Query Counts:**  
    - AI_Overview: 10–15 focused queries  
    - AI_Mode: 15–20 complex, diverse queries  
    """)
    
    # Check if API key is provided
    if 'api_key' not in st.session_state or not st.session_state.get('api_key'):
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to get started.")
        st.markdown("👈 Look at the sidebar on the left to enter your API key.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key:
            st.session_state['api_key'] = api_key
        if not api_key:
            st.info("👉 Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        
        st.subheader("Search Settings")
        search_mode = st.selectbox(
            "Search Mode",
            ["AI_Overview", "AI_Mode"],
            help="AI_Overview: 10–15 comprehensive queries\nAI_Mode: 15–20 complex, diverse queries"
        )
        
        query_intent = st.selectbox(
            "Query Intent (Optional)",
            ["", "informational", "commercial", "transactional", "navigational"]
        )
        
        industry_context = st.text_input(
            "Industry Context (Optional)",
            placeholder="e.g., Healthcare, E-commerce, Technology"
        )
        
        user_persona = st.text_area(
            "User Persona (Optional)",
            placeholder="e.g., Small business owner looking for accounting software"
        )
        
        use_cache = st.checkbox("Use Cache", value=True)
        
        st.subheader("📊 Statistics")
        if st.session_state.generation_history:
            recent = [h['queries_generated'] for h in st.session_state.generation_history[-10:]]
            avg_recent = sum(recent) / len(recent) if recent else 0
            cache_size = len(st.session_state.cache)
            
            st.metric("Total Generations", len(st.session_state.generation_history))
            st.metric("Avg Queries (Recent)", f"{avg_recent:.1f}")
            st.metric("Cache Size", cache_size)
            
            if st.button("Clear Cache"):
                st.session_state.cache = {}
                st.success("Cache cleared!")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single Query", "Batch Processing", "History & Analytics", "Compare Outputs"])
    
    with tab1:
        st.header("Single Query Generation")
        
        primary_keyword = st.text_input(
            "Primary Keyword",
            placeholder="Enter your primary keyword or query"
        )
        
        if st.button("Generate Fan-Out", type="primary") and api_key and primary_keyword:
            with st.spinner("Generating synthetic queries..."):
                generator = QueryFanOutGenerator(api_key)
                result = generator.generate_fanout(
                    primary_keyword=primary_keyword,
                    search_mode=search_mode,
                    industry_context=industry_context or None,
                    query_intent=query_intent or None,
                    user_persona=user_persona or None,
                    use_cache=use_cache
                )
                
                if result:
                    st.markdown("## 🧠 Model's Query Generation Plan")
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Target Number of Queries Decided by Model",
                            f"{len(result['synthetic_queries'])}"
                        )
                    with col2:
                        st.metric(
                            "Actual Number of Queries Generated",
                            f"{len(result['synthetic_queries'])}"
                        )
                    with col3:
                        metadata = result['generation_metadata']
                        st.metric("Generation Time", f"{metadata['generation_time_ms']}ms")
                    
                    # Display model reasoning
                    st.markdown(f"**Model's Reasoning for This Number:** {result['query_count_reasoning']}")
                    
                    st.markdown("---")
                    # Convert to DataFrame for display
                    queries_df = pd.DataFrame(result['synthetic_queries'])
                    columns_order = ['query', 'type', 'user_intent', 'reasoning']
                    if 'confidence_score' in queries_df.columns:
                        columns_order.append('confidence_score')
                    queries_df = queries_df[columns_order]
                    
                    st.dataframe(
                        queries_df,
                        use_container_width=True,
                        hide_index=False,
                        height=500
                    )
                    
                    st.markdown("---")
                    download_col1, download_col2 = st.columns(2)
                    with download_col1:
                        st.download_button(
                            "📥 Download CSV",
                            queries_df.to_csv(index=False),
                            file_name=f"fanout_{primary_keyword.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                    with download_col2:
                        st.download_button(
                            "📥 Download JSON",
                            json.dumps(result, indent=2),
                            file_name=f"fanout_{primary_keyword.replace(' ', '_')}.json",
                            mime="application/json"
                        )
    
    with tab2:
        st.header("Batch Processing")
        
        keywords_input = st.text_area(
            "Enter Keywords (one per line)",
            height=200,
            placeholder="keyword 1\nkeyword 2\nkeyword 3"
        )
        
        batch_size = st.slider("Batch Size", 1, 20, 10)
        
        if st.button("Process Batch", type="primary") and api_key and keywords_input:
            keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
            
            if keywords:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generator = QueryFanOutGenerator(api_key)
                results = []
                
                for i, keyword in enumerate(keywords):
                    status_text.text(f"Processing: {keyword}")
                    result = generator.generate_fanout(
                        primary_keyword=keyword,
                        search_mode=search_mode,
                        industry_context=industry_context or None,
                        query_intent=query_intent or None,
                        user_persona=user_persona or None,
                        use_cache=use_cache
                    )
                    if result:
                        results.append(result)
                    
                    progress_bar.progress((i + 1) / len(keywords))
                    time.sleep(1)  # Rate limiting
                
                status_text.text("Batch processing complete!")
                
                st.subheader("📊 Batch Results Summary")
                summary_data = []
                total_queries = 0
                for res in results:
                    count = res['generation_metadata']['total_queries_generated']
                    total_queries += count
                    summary_data.append({
                        'Primary Keyword': res['primary_keyword'],
                        'Queries Generated': count,
                        'Generation Time (ms)': res['generation_metadata']['generation_time_ms']
                    })
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Keywords Processed", len(results))
                with col2:
                    st.metric("Total Queries Generated", total_queries)
                with col3:
                    avg_q = total_queries / len(results) if results else 0
                    st.metric("Average Queries per Keyword", f"{avg_q:.1f}")
                
                # Show summary table
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Build DataFrame of all generated queries
                all_rows = []
                for res in results:
                    pk = res['primary_keyword']
                    for q in res['synthetic_queries']:
                        all_rows.append({
                            'primary_keyword': pk,
                            'query': q['query'],
                            'type': q['type'],
                            'user_intent': q.get('user_intent', ""),
                            'reasoning': q['reasoning'],
                            'confidence_score': q.get('confidence_score', "")
                        })
                all_queries_df = pd.DataFrame(all_rows)
                
                # Download options for batch mode
                st.markdown("---")
                download_batch_col1, download_batch_col2, download_batch_col3 = st.columns(3)
                
                with download_batch_col1:
                    st.download_button(
                        "📥 Download Summary CSV",
                        summary_df.to_csv(index=False),
                        file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with download_batch_col2:
                    st.download_button(
                        "📥 Download All Queries CSV",
                        all_queries_df.to_csv(index=False),
                        file_name=f"batch_all_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with download_batch_col3:
                    all_results = {
                        'batch_metadata': {
                            'total_keywords': len(keywords),
                            'successful_generations': len(results),
                            'timestamp': datetime.now().isoformat()
                        },
                        'results': results
                    }
                    st.download_button(
                        "📥 Download All Results JSON",
                        json.dumps(all_results, indent=2),
                        file_name=f"batch_fanout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    with tab3:
        st.header("History & Analytics")
        
        if st.session_state.generation_history:
            history_df = pd.DataFrame(st.session_state.generation_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Generations", len(history_df))
            with col2:
                avg_q = history_df['queries_generated'].mean()
                st.metric("Avg Queries per Keyword", f"{avg_q:.1f}")
            with col3:
                mode_dist = history_df['search_mode'].value_counts()
                most_used = mode_dist.index[0] if len(mode_dist) > 0 else "N/A"
                st.metric("Most Used Mode", most_used)
            with col4:
                recent_avg = history_df.tail(10)['queries_generated'].mean()
                st.metric("Recent Avg (Last 10)", f"{recent_avg:.1f}")
            
            st.subheader("Generation History")
            st.dataframe(
                history_df.sort_values('timestamp', ascending=False).head(50),
                use_container_width=True
            )
            
            if st.button("Clear History"):
                st.session_state.generation_history = []
                st.success("History cleared!")
                safe_rerun()
        else:
            st.info("No generation history yet. Start generating queries to see analytics.")
    
    with tab4:
        st.header("Compare Outputs")
        st.markdown("Upload CSV files to compare different query generation outputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Qforia Output")
            qforia_file = st.file_uploader("Upload Qforia CSV", type=['csv'], key='qforia')
            if qforia_file:
                qforia_df = pd.read_csv(qforia_file)
                st.metric("Queries Generated", len(qforia_df))
                st.dataframe(qforia_df, use_container_width=True, height=400)
        
        with col2:
            st.subheader("Your Output")
            your_file = st.file_uploader("Upload Your CSV", type=['csv'], key='yours')
            if your_file:
                your_df = pd.read_csv(your_file)
                st.metric("Queries Generated", len(your_df))
                st.dataframe(your_df, use_container_width=True, height=400)
        
        if qforia_file and your_file:
            st.subheader("📊 Comparison Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Query Count Difference",
                    f"{len(your_df) - len(qforia_df):+d}",
                    f"{((len(your_df) / len(qforia_df)) - 1) * 100:+.1f}%"
                )
            
            with col2:
                if 'query' in qforia_df.columns and 'query' in your_df.columns:
                    qforia_queries = set(qforia_df['query'].str.lower())
                    your_queries = set(your_df['query'].str.lower())
                    overlap = len(qforia_queries.intersection(your_queries))
                    st.metric("Query Overlap", f"{overlap} queries",
                             f"{(overlap / len(qforia_queries)) * 100:.1f}% of Qforia")
            
            with col3:
                if 'type' in qforia_df.columns and 'type' in your_df.columns:
                    qforia_types = qforia_df['type'].value_counts()
                    your_types = your_df['type'].value_counts()
                    type_diffs = {}
                    for t in set(qforia_types.index).union(set(your_types.index)):
                        q_count = qforia_types.get(t, 0)
                        y_count = your_types.get(t, 0)
                        type_diffs[t] = abs(q_count - y_count)
                    most_diff = max(type_diffs, key=type_diffs.get) if type_diffs else "N/A"
                    st.metric("Most Different Type", most_diff,
                             f"Δ = {type_diffs.get(most_diff, 0)}")
            
            st.subheader("Type Distribution Comparison")
            if 'type' in qforia_df.columns and 'type' in your_df.columns:
                type_comp = pd.DataFrame({
                    'Qforia': qforia_df['type'].value_counts(),
                    'Yours': your_df['type'].value_counts()
                }).fillna(0).astype(int)
                st.dataframe(type_comp, use_container_width=True)
            
            st.subheader("Unique Queries Analysis")
            if 'query' in qforia_df.columns and 'query' in your_df.columns:
                unique_qf = qforia_queries - your_queries
                unique_y = your_queries - qforia_queries
                
                uq_col1, uq_col2 = st.columns(2)
                with uq_col1:
                    st.write(f"**Unique to Qforia ({len(unique_qf)} queries):**")
                    for q in list(unique_qf)[:10]:
                        st.write(f"• {q}")
                    if len(unique_qf) > 10:
                        st.write(f"... and {len(unique_qf) - 10} more")
                
                with uq_col2:
                    st.write(f"**Unique to Yours ({len(unique_y)} queries):**")
                    for q in list(unique_y)[:10]:
                        st.write(f"• {q}")
                    if len(unique_y) > 10:
                        st.write(f"... and {len(unique_y) - 10} more")

if __name__ == "__main__":
    main()
