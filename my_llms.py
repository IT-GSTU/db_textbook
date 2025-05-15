from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def get_llm(model_name, api_key, **kwargs):
    # Общие настройки для ChatOpenAI

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=25,  # Controls the maximum burst size.
    )

    common_config = {
        "temperature": 0,
        "top_p": 0.7,
        "api_key": api_key,
        "timeout": 720,
        "max_retries": 10,
        "max_tokens": 16600,        
        "organization": "University GSTU",
        "rate_limiter": rate_limiter
    }


    if model_name in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "meta-llama/Llama-3.3-70B-Instruct"]:
        config = common_config.copy()
        config.update({            
            "base_url": "https://api-inference.huggingface.co/v1/",
        })
        config.update(kwargs)
        return ChatOpenAI(model=model_name, **config)

    elif model_name in ["gemini-2.5-pro-exp-03-25","gemini-2.0-flash"]:
        # Для ChatGoogleGenerativeAI общие настройки не применимы
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries= 10,
            max_tokens=110000,
            max_output_tokens=50000,
            **kwargs
        )
    elif model_name in ["gemma3:1b", "deepseek-r1:8b","deepseek-r1:1.5b","qwen2.5:3b"]:
        # Локальная модель с ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=0,
            **kwargs
        )
    elif model_name in ["deepseek-chat", "deepseek-reasoner"]:
        config = common_config.copy()
        config.update({
            "base_url": "https://api.deepseek.com/v1",
        })
        config.update(kwargs)
        return ChatOpenAI(model=model_name, **config)

    elif model_name in ["deepseek-ai/DeepSeek-R1"]:
        config = common_config.copy()
        config.update({
            "base_url": "https://huggingface.co/api/inference-proxy/together",
        })
        config.update(kwargs)
        return ChatOpenAI(model=model_name, **config)

    elif model_name in ["deepseek/deepseek-r1:free", "deepseek/deepseek-r1-distill-llama-70b:free","deepseek/deepseek-chat", "deepseek/deepseek-chat-v3-0324:free",
                        "google/gemini-2.5-pro-exp-03-25:free","google/gemini-2.0-pro-exp-02-05:free",
                        "qwen/qwq-32b:free", "google/gemma-3-27b-it:free","meta-llama/llama-4-scout:free"]:
        config = common_config.copy()
        config.update({
            "base_url": "https://openrouter.ai/api/v1",
        })
        config.update(kwargs)
        return ChatOpenAI(model=model_name, **config)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


