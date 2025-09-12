import gradio as gr
import pymupdf  # PyMuPDF
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import io
import json
import re

def extract_pdf_text(pdf_file):
    """Extrai texto e metadados básicos do PDF"""
    try:
        # Abrir o PDF com PyMuPDF
        doc = pymupdf.open(stream=pdf_file, filetype="pdf")
        
        # Extrair texto de todas as páginas
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n"
        
        # Extrair metadados básicos
        metadata_dict = doc.metadata
        metadata = {
            "num_pages": len(doc),
            "title": metadata_dict.get('title', 'Não disponível') if metadata_dict.get('title') else 'Não disponível',
            "author": metadata_dict.get('author', 'Não disponível') if metadata_dict.get('author') else 'Não disponível',
            "subject": metadata_dict.get('subject', 'Não disponível') if metadata_dict.get('subject') else 'Não disponível',
            "creator": metadata_dict.get('creator', 'Não disponível') if metadata_dict.get('creator') else 'Não disponível'
        }
        
        # Fechar o documento
        doc.close()
        
        return full_text, metadata
    except Exception as e:
        return None, {"error": f"Erro ao processar PDF: {str(e)}"}

def extract_references_with_llm(text):
    """Usa OpenAI para extrair e estruturar referências"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""
        Analise o texto do artigo científico abaixo e extraia APENAS a seção de referências bibliográficas.
        
        Para cada referência encontrada, extraia as seguintes informações em formato JSON:
        - authors: lista de autores
        - title: título do trabalho
        - journal: nome da revista/conferência
        - year: ano de publicação
        - volume: volume (se disponível)
        - pages: páginas (se disponível)
        - doi: DOI (se disponível)
        
        Retorne um array JSON com todas as referências encontradas.
        
        Texto do artigo:
        {text[:8000]}  # Limita o texto para evitar exceder limites da API
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        # Extrair JSON da resposta
        content = response.choices[0].message.content
        # Procurar por JSON na resposta
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            references_data = json.loads(json_match.group())
            return references_data
        else:
            return []
            
    except Exception as e:
        return [{"error": f"Erro ao processar com LLM: {str(e)}"}]

def process_pdf(pdf_file):
    """Função principal que processa o PDF e retorna resultados"""
    if pdf_file is None:
        return {"error": "Nenhum arquivo enviado"}, pd.DataFrame()
    
    # Extrair texto do PDF
    text, metadata = extract_pdf_text(pdf_file)
    
    if text is None:
        return metadata, pd.DataFrame()
    
    # Extrair referências com LLM
    references = extract_references_with_llm(text)
    
    # Converter para DataFrame
    if references and not any("error" in ref for ref in references):
        df = pd.DataFrame(references)
    else:
        df = pd.DataFrame({"Erro": ["Não foi possível extrair referências"]})
    
    return metadata, df

def create_interface():
    """Cria a interface Gradio"""
    with gr.Blocks(title="Extrator de Referências") as interface:
        gr.Markdown("# 📚 Extrator de Referências de Artigos Científicos")
        gr.Markdown("Faça upload de um PDF de artigo científico para extrair automaticamente a lista de referências.")
        
        with gr.Row():
            pdf_input = gr.File(
                label="📄 Upload do PDF",
                file_types=[".pdf"],
                type="binary"
            )
        
        extract_btn = gr.Button("🔍 Extrair Referências", variant="primary")
        
        with gr.Row():
            with gr.Column():
                metadata_output = gr.JSON(label="📋 Metadados do Artigo")
            with gr.Column():
                references_output = gr.Dataframe(
                    label="📖 Lista de Referências",
                    row_count=(10,'dynamic'),
                    show_copy_button=True,
                    show_fullscreen_button=True,
                    wrap=True
                )
        
        extract_btn.click(
            process_pdf,
            inputs=[pdf_input],
            outputs=[metadata_output, references_output]
        )
    
    return interface

def main():
    load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
    
    # Verificar se a chave da API está configurada
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  AVISO: Chave da API OpenAI não encontrada!")
        print("Crie um arquivo .env com: OPENAI_API_KEY=sua_chave_aqui")
    
    interface = create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
