"""
Test script for all LLM clients to diagnose API issues.
Run: python test_llm_clients.py
"""
import json
import httpx
from config import (
    EmbeddingConfig, RerankerConfig, LLMConfig,
    MOONSHOT_API_KEY, MOONSHOT_BASE_URL, MODEL_NAME
)

print("=" * 70)
print("LLM Clients Diagnostic Test")
print("=" * 70)

# Test 1: Legacy call_moonshot_json
print("\n[TEST 1] Testing legacy call_moonshot_json...")
print("-" * 70)

try:
    url = f"{MOONSHOT_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {MOONSHOT_API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Return only valid JSON."},
            {"role": "user", "content": "Say hello in JSON format with a 'message' field."}
        ],
        "temperature": 0.3
    }
    
    print(f"URL: {url}")
    print(f"Model: {MODEL_NAME}")
    print(f"Headers: {headers}")
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        response.raise_for_status()
        result = response.json()
        print(f"Full Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        content = result['choices'][0]['message']['content']
        print(f"\nExtracted Content: {content}")
        print(f"Content Type: {type(content)}")
        print(f"Content Length: {len(content)}")
        
        # Try to parse as JSON
        try:
            import re
            clean_content = re.sub(r'^```json\s*|```$', '', content.strip(), flags=re.MULTILINE)
            print(f"Cleaned Content: {clean_content}")
            parsed = json.loads(clean_content)
            print(f"✓ Successfully parsed JSON: {parsed}")
        except Exception as e:
            print(f"✗ JSON parsing failed: {e}")
            print(f"Raw content: {repr(content)}")
            
except Exception as e:
    print(f"✗ Request failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: EmbeddingClient
print("\n\n[TEST 2] Testing EmbeddingClient...")
print("-" * 70)

try:
    from llm_client import EmbeddingClient
    
    embedding_client = EmbeddingClient()
    print(f"API URL: {embedding_client.api_url}")
    print(f"Model: {embedding_client.model_name}")
    print(f"API Key: {embedding_client.api_key[:10]}...")
    
    # Test single embedding
    test_text = "This is a test sentence for embedding."
    print(f"\nTest Text: {test_text}")
    
    embedding = embedding_client.embed_text(test_text)
    print(f"✓ Embedding generated successfully")
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    
except Exception as e:
    print(f"✗ EmbeddingClient failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: RerankerClient
print("\n\n[TEST 3] Testing RerankerClient...")
print("-" * 70)

try:
    from llm_client import RerankerClient
    
    reranker_client = RerankerClient()
    print(f"API URL: {reranker_client.api_url}")
    print(f"Model: {reranker_client.model_name}")
    print(f"API Key: {reranker_client.api_key[:10]}...")
    
    # Test reranking
    query = "What is Python?"
    documents = [
        "Python is a programming language",
        "JavaScript is also a programming language",
        "Python is used for data science"
    ]
    
    print(f"\nQuery: {query}")
    print(f"Documents: {len(documents)}")
    
    results = reranker_client.rerank(query, documents)
    print(f"✓ Reranking successful")
    print(f"  Results: {len(results)}")
    for i, result in enumerate(results[:3]):
        print(f"  [{i}] Score: {result['relevance_score']:.4f}, Text: {result['text'][:50]}...")
    
except Exception as e:
    print(f"✗ RerankerClient failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Direct LLM call with different response format
print("\n\n[TEST 4] Testing LLM with text response (non-JSON)...")
print("-" * 70)

try:
    url = f"{MOONSHOT_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {MOONSHOT_API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ],
        "temperature": 0.3
    }
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"✓ Text response: {content}")
        
except Exception as e:
    print(f"✗ Request failed: {e}")

# Test 5: Check if issue is with JSON response format
print("\n\n[TEST 5] Testing JSON response parsing...")
print("-" * 70)

try:
    url = f"{MOONSHOT_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {MOONSHOT_API_KEY}"}
    
    # Ask LLM to return JSON
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You must respond with valid JSON only. No markdown, no code blocks, just pure JSON."
            },
            {
                "role": "user",
                "content": 'Return this exact JSON: {"summary": "This is a test summary", "status": "success"}'
            }
        ],
        "temperature": 0.0
    }
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        print(f"Raw content: {repr(content)}")
        print(f"Content starts with: {content[:50] if content else 'EMPTY'}")
        
        # Try parsing
        try:
            parsed = json.loads(content)
            print(f"✓ Direct JSON parsing successful: {parsed}")
        except:
            # Try cleaning
            import re
            clean_content = re.sub(r'^```json\s*|```$', '', content.strip(), flags=re.MULTILINE)
            clean_content = clean_content.strip()
            print(f"Cleaned content: {repr(clean_content)}")
            
            try:
                parsed = json.loads(clean_content)
                print(f"✓ JSON parsing successful after cleaning: {parsed}")
            except Exception as e2:
                print(f"✗ Still failed after cleaning: {e2}")
                
except Exception as e:
    print(f"✗ Request failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Diagnostic Complete!")
print("=" * 70)
