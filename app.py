import gradio as gr
import pymupdf  # PyMuPDF
import pandas as pd
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv
import io
import json
import re

class Reference(BaseModel):
    authors: List[str]
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None

class ReferencesResponse(BaseModel):
    references: List[Reference]

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

def extract_references_with_llm(text, model_name):
    """Usa Pydantic AI com diferentes modelos para extrair e estruturar referências"""
    try:
        # Determinar se é modelo Google ou OpenAI
        if model_name.startswith('gemini'):
            # Configurar a API key do Google
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            api_key = os.getenv("GOOGLE_API_KEY")
        else:
            # Usar OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return [{"error": f"Chave da API não encontrada para o modelo {model_name}"}]
        
        # Criar o agente Pydantic AI
        agent = Agent(
            model_name,
            result_type=ReferencesResponse,
            system_prompt="""
            Você é um especialista em análise de artigos científicos. 
            Sua tarefa é identificar e extrair APENAS a seção de referências bibliográficas do texto fornecido.
            
            Para cada referência encontrada, extraia:
            - authors: lista completa de autores
            - title: título completo do trabalho
            - journal: nome da revista/conferência/editora
            - year: ano de publicação
            - volume: volume (se disponível)
            - pages: páginas (se disponível)
            - doi: DOI (se disponível)
            
            Seja preciso e extraia apenas referências válidas e completas.
            """
        )
        
        # Ajustar limite de texto baseado no modelo
        if model_name.startswith('gemini'):
            limited_text = text[:150000]  # Gemini tem limite maior
        else:
            limited_text = text[:50000]   # OpenAI tem limite menor
        
        # Executar o agente
        result = agent.run_sync(f"Extraia as referências bibliográficas do seguinte texto de artigo científico:\n\n{limited_text}")
        
        # Converter para lista de dicionários para compatibilidade com DataFrame
        references_list = []
        for ref in result.data.references:
            references_list.append({
                "authors": ", ".join(ref.authors) if ref.authors else "",
                "title": ref.title,
                "journal": ref.journal or "",
                "year": ref.year or "",
                "volume": ref.volume or "",
                "pages": ref.pages or "",
                "doi": ref.doi or ""
            })
        
        return references_list
            
    except Exception as e:
        return [{"error": f"Erro ao processar com LLM ({model_name}): {str(e)}"}]

def process_pdf(pdf_file, model_name):
    """Função principal que processa o PDF e retorna resultados"""
    if pdf_file is None:
        return {"error": "Nenhum arquivo enviado"}, pd.DataFrame()
    
    # Extrair texto do PDF
    text, metadata = extract_pdf_text(pdf_file)
    
    if text is None:
        return metadata, pd.DataFrame()
    
    # Adicionar modelo selecionado aos metadados
    metadata["modelo_usado"] = model_name
    
    # Extrair referências com LLM
    references = extract_references_with_llm(text, model_name)
    
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
            with gr.Column():
                pdf_input = gr.File(
                    label="📄 Upload do PDF",
                    file_types=[".pdf"],
                    type="binary"
                )
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=[
                        "gemini-2.5-flash-lite",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gpt-4o",
                        "gpt-o3-mini",
                        "gpt-4.1"
                    ],
                    value="gemini-2.5-flash-lite",
                    label="🤖 Modelo de IA",
                    info="Selecione o modelo para extrair as referências"
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
            inputs=[pdf_input, model_dropdown],
            outputs=[metadata_output, references_output]
        )
    
    return interface

def main():
    load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
    
    # Verificar se as chaves das APIs estão configuradas
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not google_key and not openai_key:
        print("⚠️  AVISO: Nenhuma chave de API encontrada!")
        print("Configure pelo menos uma das seguintes no arquivo .env:")
        print("- GEMINI_API_KEY=sua_chave_do_google")
        print("- OPENAI_API_KEY=sua_chave_da_openai")
    elif not google_key:
        print("ℹ️  Apenas OpenAI configurado. Modelos Gemini não funcionarão.")
    elif not openai_key:
        print("ℹ️  Apenas Google configurado. Modelos OpenAI não funcionarão.")
    
    interface = create_interface()
    interface.launch(share=False)

if __name__ == "__main__":
    main()
