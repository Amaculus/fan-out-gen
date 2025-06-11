#!/usr/bin/env python3
# pages/ai_overview_multi.py  ── Streamlit version
import os
import time
import random
import json
from html import unescape
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
from requests.exceptions import RequestException

# LLM SDKs
from openai import OpenAI
from anthropic import Anthropic

# SerpAPI
from serpapi.google_search import GoogleSearch

# Configure page
st.set_page_config(
    page_title="AI Overview Multi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ai_overview_results' not in st.session_state:
    st.session_state.ai_overview_results = []

DEFAULT_TIMEOUT = 15  # seconds

# ── helper: flatten text_blocks → plain text ──────────────────────────────────
def _blocks_to_text(blocks: list[dict]) -> str:
    """
    Turn ai_overview.text_blocks → plain text.
    Handles paragraph / list / table blocks and skips unknown types.
    Falls back gracefully if 'snippet' is missing.
    """
    def _extract(d: dict) -> str | None:
        for key in ("snippet", "text", "title", "content"):
            if key in d and d[key]:
                return d[key]
        return None

    out: list[str] = []

    for b in blocks:
        match b.get("type"):
            case "paragraph":
                if txt := _extract(b):
                    out.append(txt)

            case "list":
                for item in b.get("list", []):
                    if txt := _extract(item):
                        out.append(f"• {txt}")

            case "table":      # optional: flatten tables row-by-row
                rows = b.get("table", {}).get("rows", [])
                for row in rows:
                    cells = [_extract(c) or "" for c in row.get("cells", [])]
                    line  = " | ".join(cells).strip(" |")
                    if line:
                        out.append(line)

            case _:
                continue       # skip unknown block types

    return "\n".join(unescape(x) for x in out).strip()

# ── helper: fetch AI Overview once ────────────────────────────────────────────
def fetch_ai_overview_once(keyword: str, serpapi_key: str, gl: str = "us") -> str | None:
    """Single attempt to pull an Overview; returns plain text or None."""
    try:
        # 1️⃣ Normal search
        search = GoogleSearch({
            "engine": "google",
            "q": keyword,
            "api_key": serpapi_key,
            "hl": "en",
            "gl": gl,
            "no_cache": "true",
        })
        search.timeout = DEFAULT_TIMEOUT
        root = search.get_dict()

        aio = root.get("ai_overview")
        if not aio:
            return None

        blocks = aio.get("text_blocks")
        if blocks:
            return _blocks_to_text(blocks)

        # 2️⃣ streamed payload
        token = aio.get("page_token")
        if not token:
            return None

        full_search = GoogleSearch({
            "engine": "google_ai_overview",
            "page_token": token,
            "api_key": serpapi_key,
            "no_cache": "true",
        })
        full_search.timeout = DEFAULT_TIMEOUT
        full = full_search.get_dict()

        blocks = full.get("ai_overview", {}).get("text_blocks", [])
        return _blocks_to_text(blocks) if blocks else None
        
    except RequestException as e:
        st.warning(f"SerpApi timeout for '{keyword}': {e}")
        return None
    except Exception as e:
        st.error(f"Error fetching AI Overview for '{keyword}': {e}")
        return None

# ── reducer: merge N snapshots into one consensus text ────────────────────────
def reduce_snapshots(snaps: list[str]) -> str:
    """
    Very simple strategy: pick the *longest* distinct version.
    You can swap for majority-vote, n-gram fusion, etc.
    """
    uniq = list(dict.fromkeys(snaps))          # preserve order, dedupe
    if not uniq:
        return ""
    longest = max(uniq, key=len)
    return longest

# ── shared rewrite prompt ─────────────────────────────────────────────────────
def build_prompt(text: str) -> str:
    return (
        "Rewrite the following Google AI Overview in a fresh, human voice. "
        "Keep all key facts, avoid plagiarism, limit to 1-3 short paragraphs.\n\n"
        f"AI Overview:\n{text}\n\nRewritten version:"
    )

# ── model-specific wrappers ────────────────────────────────────────────────────
def rewrite_openai(text: str, api_key: str, temperature: float = 0.7) -> str:
    try:
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_prompt(text)}],
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return f"Error generating OpenAI rewrite: {e}"

def rewrite_claude(text: str, api_key: str, temperature: float = 0.7) -> str:
    try:
        anthropic_client = Anthropic(api_key=api_key)
        r = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            temperature=temperature,
            messages=[{"role": "user", "content": build_prompt(text)}],
        )
        return r.content[0].text.strip()
    except Exception as e:
        st.error(f"Anthropic API error: {e}")
        return f"Error generating Claude rewrite: {e}"

# ── main processing function ──────────────────────────────────────────────────
def process_keywords(
    keywords: List[str], 
    runs: int, 
    delay: float,
    serpapi_key: str,
    openai_key: str,
    anthropic_key: str,
    gl: str = "us"
) -> List[Dict]:
    """Process keywords and return results"""
    results = []
    
    # Create progress tracking
    total_operations = len(keywords) * (runs + 2)  # snapshots + 2 rewrites per keyword
    progress_bar = st.progress(0)
    status_container = st.container()
    current_operation = 0
    
    for kw_idx, kw in enumerate(keywords):
        with status_container:
            st.info(f"🔍 Processing keyword: **{kw}** ({kw_idx + 1}/{len(keywords)})")
        
        snapshots: List[str] = []
        
        # Collect snapshots
        snapshot_container = st.expander(f"📸 Snapshot Collection for '{kw}'", expanded=True)
        
        for i in range(1, runs + 1):
            with snapshot_container:
                snapshot_status = st.empty()
                snapshot_status.text(f"Collecting snapshot {i}/{runs}...")
            
            snap = fetch_ai_overview_once(kw, serpapi_key, gl)
            
            with snapshot_container:
                if snap:
                    st.success(f"✓ Snapshot {i}: Captured ({len(snap)} chars)")
                    snapshots.append(snap)
                else:
                    st.warning(f"— Snapshot {i}: No AI Overview found")
            
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
            
            if i < runs:  # Don't delay after the last snapshot
                time.sleep(delay + random.uniform(0.3, 0.8))
        
        # Process results for this keyword
        keyword_result = {
            'keyword': kw,
            'snapshots_collected': len(snapshots),
            'total_runs': runs,
            'consensus_text': '',
            'openai_rewrite': '',
            'claude_rewrite': '',
            'timestamp': datetime.now().isoformat()
        }
        
        if not snapshots:
            with status_container:
                st.warning(f"⚠️ No AI Overview found for '{kw}' - skipping rewrites")
            keyword_result['error'] = 'No AI Overview found'
            results.append(keyword_result)
            current_operation += 2  # Skip the rewrite operations
            progress_bar.progress(current_operation / total_operations)
            continue
        
        # Generate consensus
        consensus = reduce_snapshots(snapshots)
        keyword_result['consensus_text'] = consensus
        
        # Display consensus
        consensus_container = st.expander(f"🗂️ Consensus AI Overview for '{kw}'", expanded=True)
        with consensus_container:
            st.markdown("**Consolidated AI Overview (input to LLMs):**")
            st.text_area("", value=consensus, height=150, disabled=True, key=f"consensus_{kw_idx}")
        
        # Generate rewrites
        rewrite_container = st.expander(f"✏️ LLM Rewrites for '{kw}'", expanded=True)
        
        with rewrite_container:
            col1, col2 = st.columns(2)
            
            # OpenAI rewrite
            with col1:
                st.markdown("**GPT-4o-mini Rewrite:**")
                if openai_key:
                    with st.spinner("Generating OpenAI rewrite..."):
                        openai_result = rewrite_openai(consensus, openai_key)
                        keyword_result['openai_rewrite'] = openai_result
                        st.text_area("", value=openai_result, height=200, disabled=True, key=f"openai_{kw_idx}")
                else:
                    st.warning("OpenAI API key not provided")
                    keyword_result['openai_rewrite'] = "No API key provided"
            
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
            
            # Claude rewrite  
            with col2:
                st.markdown("**Claude Sonnet Rewrite:**")
                if anthropic_key:
                    with st.spinner("Generating Claude rewrite..."):
                        claude_result = rewrite_claude(consensus, anthropic_key)
                        keyword_result['claude_rewrite'] = claude_result
                        st.text_area("", value=claude_result, height=200, disabled=True, key=f"claude_{kw_idx}")
                else:
                    st.warning("Anthropic API key not provided")
                    keyword_result['claude_rewrite'] = "No API key provided"
            
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
        
        results.append(keyword_result)
        
        # Add some spacing between keywords
        st.markdown("---")
    
    return results

# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.title("🤖 AI Overview Multi")
    st.markdown("""
    Capture multiple Google AI Overview snapshots and generate fresh rewrites using different LLMs.
    
    **How it works:**
    1. Fetches multiple AI Overview snapshots for each keyword
    2. Consolidates them into a consensus version
    3. Generates fresh rewrites using GPT-4o-mini and Claude Sonnet
    """)
    
    # Check for API keys
    missing_keys = []
    if not st.session_state.get('serpapi_key'):
        missing_keys.append("SerpAPI")
    if not st.session_state.get('openai_key') and not st.session_state.get('anthropic_key'):
        missing_keys.append("at least one LLM API (OpenAI or Anthropic)")
    
    if missing_keys:
        st.warning(f"⚠️ Please configure your API keys in the sidebar: {', '.join(missing_keys)}")
        st.markdown("👈 Look at the sidebar on the left to enter your API keys.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # SerpAPI Key (required)
        serpapi_key = st.text_input(
            "SerpAPI Key *", 
            type="password", 
            value=st.session_state.get('serpapi_key', ''),
            help="Required for fetching Google AI Overviews"
        )
        if serpapi_key:
            st.session_state['serpapi_key'] = serpapi_key
        if not serpapi_key:
            st.info("👉 Get your API key from [SerpAPI](https://serpapi.com/)")
        
        st.markdown("---")
        st.subheader("LLM APIs (Optional)")
        st.caption("Provide at least one for rewrites")
        
        # OpenAI Key
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.get('openai_key', ''),
            help="For GPT-4o-mini rewrites"
        )
        if openai_key:
            st.session_state['openai_key'] = openai_key
        
        # Anthropic Key
        anthropic_key = st.text_input(
            "Anthropic API Key", 
            type="password",
            value=st.session_state.get('anthropic_key', ''),
            help="For Claude Sonnet rewrites"
        )
        if anthropic_key:
            st.session_state['anthropic_key'] = anthropic_key
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        # Geographic location
        gl = st.selectbox(
            "Geographic Location",
            ["us", "uk", "ca", "au", "de", "fr", "jp"],
            help="Google search geographic location"
        )
        
        # Number of snapshots
        runs = st.number_input(
            "Snapshots per keyword", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="More snapshots = better consensus but slower"
        )
        
        # Delay between requests
        delay = st.number_input(
            "Delay between snapshots (seconds)", 
            min_value=0.5, 
            max_value=10.0, 
            value=2.0, 
            step=0.5,
            help="Prevents rate limiting"
        )
        
        st.markdown("---")
        st.subheader("📊 Session Stats")
        if st.session_state.ai_overview_results:
            total_keywords = len(st.session_state.ai_overview_results)
            successful = len([r for r in st.session_state.ai_overview_results if not r.get('error')])
            st.metric("Keywords Processed", total_keywords)
            st.metric("Successful", successful)
            st.metric("Success Rate", f"{(successful/total_keywords)*100:.1f}%")
            
            if st.button("Clear History"):
                st.session_state.ai_overview_results = []
                st.success("History cleared!")
                st.rerun()
    
    # Main content
    tab1, tab2 = st.tabs(["Process Keywords", "Results History"])
    
    with tab1:
        st.header("Process Keywords")
        
        # Keywords input
        keywords_input = st.text_area(
            "Enter keywords (one per line)",
            height=150,
            placeholder="artificial intelligence\nmachine learning\nneural networks",
            help="Each line will be processed as a separate keyword"
        )
        
        # Parse keywords
        keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
        
        if keywords:
            st.info(f"📝 **{len(keywords)} keywords** ready for processing")
            
            # Estimate processing time
            estimated_time = len(keywords) * (runs * (delay + 1) + 10)  # rough estimate
            st.caption(f"⏱️ Estimated processing time: ~{estimated_time/60:.1f} minutes")
        
        # Process button
        can_process = (
            serpapi_key and 
            keywords and 
            (openai_key or anthropic_key)
        )
        
        if st.button("🚀 Process Keywords", type="primary", disabled=not can_process):
            if not can_process:
                st.error("Please provide SerpAPI key, keywords, and at least one LLM API key")
            else:
                # Start processing
                with st.container():
                    st.markdown("### 🔄 Processing Results")
                    
                    results = process_keywords(
                        keywords=keywords,
                        runs=runs,
                        delay=delay,
                        serpapi_key=serpapi_key,
                        openai_key=openai_key or "",
                        anthropic_key=anthropic_key or "",
                        gl=gl
                    )
                    
                    # Store results in session state
                    st.session_state.ai_overview_results.extend(results)
                    
                    # Success summary
                    successful = len([r for r in results if not r.get('error')])
                    st.success(f"✅ Processing complete! {successful}/{len(results)} keywords successful")
                    
                    # Download options
                    if results:
                        st.markdown("### 📥 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV download
                            df_data = []
                            for r in results:
                                df_data.append({
                                    'keyword': r['keyword'],
                                    'snapshots_collected': r['snapshots_collected'],
                                    'consensus_text': r['consensus_text'],
                                    'openai_rewrite': r['openai_rewrite'],
                                    'claude_rewrite': r['claude_rewrite'],
                                    'timestamp': r['timestamp']
                                })
                            df = pd.DataFrame(df_data)
                            csv = df.to_csv(index=False)
                            
                            st.download_button(
                                "📊 Download CSV",
                                csv,
                                file_name=f"ai_overview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON download
                            json_data = {
                                'metadata': {
                                    'total_keywords': len(keywords),
                                    'successful_generations': successful,
                                    'settings': {
                                        'runs': runs,
                                        'delay': delay,
                                        'geographic_location': gl
                                    },
                                    'timestamp': datetime.now().isoformat()
                                },
                                'results': results
                            }
                            
                            st.download_button(
                                "📄 Download JSON",
                                json.dumps(json_data, indent=2),
                                file_name=f"ai_overview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
    
    with tab2:
        st.header("Results History")
        
        if st.session_state.ai_overview_results:
            # Summary metrics
            total_results = len(st.session_state.ai_overview_results)
            successful_results = len([r for r in st.session_state.ai_overview_results if not r.get('error')])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Keywords", total_results)
            with col2:
                st.metric("Successful", successful_results)
            with col3:
                success_rate = (successful_results / total_results) * 100 if total_results > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Results table
            st.subheader("📋 Results Summary")
            summary_data = []
            for r in st.session_state.ai_overview_results:
                summary_data.append({
                    'Keyword': r['keyword'],
                    'Snapshots': r['snapshots_collected'],
                    'Status': '✅ Success' if not r.get('error') else f"❌ {r.get('error', 'Error')}",
                    'Consensus Length': len(r.get('consensus_text', '')),
                    'Timestamp': r['timestamp'][:19].replace('T', ' ')  # Format timestamp
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Detailed results
            st.subheader("🔍 Detailed Results")
            for i, result in enumerate(st.session_state.ai_overview_results):
                with st.expander(f"📋 {result['keyword']} - {result['timestamp'][:19]}"):
                    if result.get('error'):
                        st.error(f"Error: {result['error']}")
                        continue
                    
                    st.write(f"**Snapshots collected:** {result['snapshots_collected']}")
                    
                    if result.get('consensus_text'):
                        st.markdown("**Consensus AI Overview:**")
                        st.text_area("", value=result['consensus_text'], height=100, disabled=True, key=f"hist_consensus_{i}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result.get('openai_rewrite'):
                            st.markdown("**GPT-4o-mini Rewrite:**")
                            st.text_area("", value=result['openai_rewrite'], height=150, disabled=True, key=f"hist_openai_{i}")
                    
                    with col2:
                        if result.get('claude_rewrite'):
                            st.markdown("**Claude Sonnet Rewrite:**")
                            st.text_area("", value=result['claude_rewrite'], height=150, disabled=True, key=f"hist_claude_{i}")
        
        else:
            st.info("No results yet. Process some keywords in the first tab to see results here.")

if __name__ == "__main__":
    main()