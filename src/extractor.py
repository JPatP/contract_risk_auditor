import os
import json
from anyio import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env 

# 1. Define the exact JSON structure you expect from the model
class ContractAuditor(BaseModel):
    has_auto_renewal: bool = Field(description="Indicates if the contract has an automatic renewal clause.")
    termination_notice_period: int = Field(description="The notice period required for termination in days.")
    has_unlimited_liability: bool = Field(description="Indicates if the contract includes an unlimited liability clause.")

    evidence_text: str = Field(description="Textual evidence from the contract supporting the above findings.")

def extract_contract_info(contract_text: str) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 2. System prompt to guide the model's behavior to prevent hallucination
    system_prompt = """ You are an expert legal AI auditor. 
    Your job is to extract specific clauses from the provided contract text.
    RULES: 
    - If a clause is NOT mentioned, return False for booleans or 0 for integers.
    - You MUST extract the exact sentence for 'evidence_text' that supports your findings.
    - NEVER hallucinate information. If a clause is not present, return the default values.
    """

    # 3. Call the API with Structured Outputs
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the contract text: {contract_text}"} 
        ],
        # This tells the model to return the output in the structure defined by the ContractAuditor Pydantic model
        response_format=ContractAuditor 
    )
    # 4. Return the parsed output as a dictionary
    return response.choices[0].message.parsed.model_dump()  

if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = Path(__file__).resolve().parent
    
    # Define the path to the contract file 
    filepath = current_dir / "../data/contract_2_risky.txt"

    # Read contract 2 (the risky one)
    with open("../data/contract_2_risky.txt", "r") as file:
        contract_text = file.read()

    result = extract_contract_info(contract_text) # Extract contract information using the defined function
    print(json.dumps(result, indent=2)) # Print the result in a readable JSON format