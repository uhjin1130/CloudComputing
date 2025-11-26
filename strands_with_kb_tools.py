from strands import Agent, tool
from strands_tools import http_request, retrieve
import sys
import os
import io

# Define a paper analysis system prompt
PAPER_AGENT_PROMPT = """당신은 논문 도우미 입니다. 논문 내용을 분석하여 질문에 답변해주세요.

질문에 답변하기 위해 다음 순서로 도구를 사용하세요:

1. **retrieve 도구를 우선적으로 사용하세요**: 내가 수집한 Knowledge Base에 저장된 논문을 검색할 때는 반드시 retrieve 도구를 사용하세요. 질문과 관련된 논문 내용을 찾기 위해 retrieve 도구를 먼저 호출하세요.

2. **인터넷 검색**: Knowledge Base에서 충분한 정보를 찾을 수 없을 때만 http_request 도구를 사용하여 인터넷에서 추가 정보를 검색하세요.

중요: 논문 관련 질문이 있을 때는 항상 먼저 retrieve 도구를 사용하여 Knowledge Base에서 관련 논문을 검색하세요.
"""

@tool
def get_paper_metadata(keyword: str) -> dict:
    """논문 메타데이터를 조회합니다. 키워드나 제목으로 논문 정보를 검색합니다.

    Args:
        keyword: 검색할 논문의 키워드 또는 제목
    """
    # 예시 논문 데이터베이스 (실제로는 외부 API나 데이터베이스를 사용)
    paper_database = {
        "transformer": {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "venue": "NeurIPS",
            "keywords": ["transformer", "attention", "neural machine translation"]
        },
        "bert": {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Devlin et al."],
            "year": 2018,
            "venue": "NAACL",
            "keywords": ["bert", "transformer", "language model", "pre-training"]
        },
        "gpt": {
            "title": "Language Models are Unsupervised Multitask Learners",
            "authors": ["Radford et al."],
            "year": 2019,
            "venue": "OpenAI",
            "keywords": ["gpt", "language model", "unsupervised learning"]
        }
    }
    
    # 키워드로 논문 검색 (간단한 예시)
    keyword_lower = keyword.lower()
    for key, paper in paper_database.items():
        if key in keyword_lower or any(kw in keyword_lower for kw in paper["keywords"]):
            return paper
    
    return {"error": f"'{keyword}'에 해당하는 논문을 찾을 수 없습니다."}


def safe_input(prompt: str) -> str:
    """UTF-8 인코딩 오류를 안전하게 처리하는 input 함수."""
    try:
        # 먼저 일반 input 시도
        return input(prompt).strip()
    except UnicodeDecodeError:
        # 인코딩 오류 발생 시 재시도
        try:
            # stdin을 UTF-8로 재설정
            if hasattr(sys.stdin, 'buffer'):
                sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
            return input(prompt).strip()
        except (UnicodeDecodeError, UnicodeError):
            # 그래도 실패하면 raw bytes로 읽기
            try:
                sys.stdout.write(prompt)
                sys.stdout.flush()
                line = sys.stdin.buffer.readline()
                return line.decode('utf-8', errors='replace').strip()
            except Exception:
                raise


def main():
    """Main function to run the paper analysis agent as a script."""
    # Set Knowledge Base ID for retrieve tool
    # The retrieve tool uses KNOWLEDGE_BASE_ID environment variable as default
    # or can accept knowledgeBaseId parameter directly when called
    kb_id = os.environ.get("KNOWLEDGE_BASE_ID", "YOUR KB ID")
    os.environ["KNOWLEDGE_BASE_ID"] = kb_id
    
    paper_agent = Agent(
        model="us.amazon.nova-lite-v1:0",
        system_prompt=PAPER_AGENT_PROMPT,
        tools=[get_paper_metadata, http_request, retrieve]
    )
    
    # Command line argument이 있으면 한 번만 실행하고 종료
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        try:
            response = paper_agent(prompt)
            print(response)
        except UnicodeDecodeError as e:
            print(f"인코딩 오류가 발생했습니다: {e}")
            print("응답을 처리하는 중 문제가 발생했습니다.")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
        return
    
    # 대화형 모드: 사용자가 "종료" 또는 "exit"를 입력할 때까지 계속 실행
    print("논문 분석 에이전트를 시작합니다. 종료하려면 '종료' 또는 'exit'를 입력하세요.\n")
    
    while True:
        try:
            prompt = safe_input("논문에 대한 질문을 입력하세요: ")
            
            # 종료 조건 확인
            if prompt.lower() in ['종료', 'exit', 'quit', 'q']:
                print("논문 분석 에이전트를 종료합니다.")
                break
            
            # 빈 입력 처리
            if not prompt:
                print("질문을 입력해주세요.\n")
                continue
            
            # 에이전트 실행
            try:
                response = paper_agent(prompt)
                print(f"\n{response}\n")
            except UnicodeDecodeError as e:
                print(f"\n인코딩 오류가 발생했습니다: {e}")
                print("응답을 처리하는 중 문제가 발생했습니다. 다시 시도해주세요.\n")
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}\n")
                
        except KeyboardInterrupt:
            print("\n\n논문 분석 에이전트를 종료합니다.")
            break
        except EOFError:
            print("\n\n논문 분석 에이전트를 종료합니다.")
            break


if __name__ == "__main__":
    main()