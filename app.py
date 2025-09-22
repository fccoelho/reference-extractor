import gradio as gr
import pymupdf  # PyMuPDF
import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv
import io
import json
import re

# Padrões globais de regex para extração de referências
REFERENCE_PATTERNS = [
    # Padrão 0: Referências numeradas com autores múltiplos (formato: Número. Autores. Título. Journal info (ano).)
    r'^\d+\.\s*([A-Z][A-Za-z\s,&.-]+?(?:\s&\s[A-Z][A-Za-z\s,&.-]+?)*)\.\s*([^.]+?)\.\s*([^.]+?)\s+(\d+),?\s*([^(]*?)\s*\((\d{4})\)',
    
    # Padrão 1: Autor(es). (Ano). Título. Journal/Editora.
    r'^([A-Z][A-Za-z\s,&.-]+?)\.\s*\((\d{4}[a-z]?)\)\.\s*([^.]+?)\.\s*([^.]+?)\.?\s*$',
    
    # Padrão 2: Referências numeradas [1] Autor... ano Título. Journal doi:...
    r'^\[\d+\]\s*([A-Z][A-Za-z\s,&.-]+?)\s+(\d{4})\s+([^.]+?)\.\s*([^.]+?)(?:\s+doi:([^\s.]+))?\.?\s*$',
    
    # Padrão 3: Autor, A. (Ano). Título. Journal.
    r'^([A-Z][A-Za-z\s,&.-]+?)\s+\((\d{4}[a-z]?)\)[.,]\s*([^.]+?)[.,]\s*([^.]+?)\.?\s*$',
    
    # Padrão 4: Autor et al. (Ano) Título. Journal
    r'^([A-Z][A-Za-z\s,&.-]*?et\s+al\.?)\s*\((\d{4}[a-z]?)\)[.,]?\s*([^.]+?)[.,]\s*([^.]+?)\.?\s*$',
    
    # Padrão 5: Sobrenome, Nome (Ano). Título. Journal.
    r'^([A-Z][a-z]+,\s*[A-Z][A-Za-z\s,&.-]*?)\.\s*\((\d{4}[a-z]?)\)\.\s*([^.]+?)\.\s*([^.]+?)\.?\s*$',
    
    # Padrão 6: Múltiplos autores com &
    r'^([A-Z][A-Za-z\s,&.-]+?&[A-Za-z\s,&.-]+?)\.\s*\((\d{4}[a-z]?)\)\.\s*([^.]+?)\.\s*([^.]+?)\.?\s*$',
    
    # Padrão 7: Referências numeradas [número] Autor: Título, Editora (ano)
    r'^\[\d+\]\s*([A-Z][A-Za-z\s,&.-]+?):\s*([^,]+?),\s*([^(]+?)\s*\((\d{4})\)',

    # Padrão 8: Referências numeradas com DOI opcional
r"""
^              # início de linha (após possível marcador de ordem)
(?:\d+\.\s*)?  # número da referência (opcional), seguido de ponto e espaço
(?P<autores>   # grupo 'autores'
  [^\.]+?      # tudo antes do primeiro ponto final (não guloso)
)\.\s+
(?P<titulo>    # grupo 'titulo'
  [^\n\.]+     # até o próximo ponto final ou quebra de linha
)\.
\s*
(?P<journal>   # grupo 'journal'
  [^\n;]+      # até o próximo ponto e vírgula (ou quebra de linha)
)
[;,]?\s*
(?P<ano>       # grupo 'ano'
  \d{4}        # 4 dígitos (ano)
)
(?:;[^\n]*?)?  # volume, issue, páginas (opcional, não capturado)
(?:\n+         # nova linha(s), captura DOI opcional
  (?P<doi> https?://doi\.org/[^\s]+ )
)?             # DOI pode estar na linha de baixo ou ausente
"""
]

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
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            api_key = os.getenv("GEMINI_API_KEY")
        else:
            # Usar OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return [{"error": f"Chave da API não encontrada para o modelo {model_name}"}]
        
        # Criar o agente Pydantic AI
        agent = Agent(
            model_name,
            model_settings=ModelSettings(
                timeout=30,
            ),
            output_type=ReferencesResponse,
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
            
            Seja preciso e extraia referências completas.
            """
        )
        
        # Ajustar limite de texto baseado no modelo
        if model_name.startswith('gemini'):
            limited_text = text[:1500000]  # Gemini tem limite maior
        else:
            limited_text = text[:500000]   # OpenAI tem limite menor
        
        # Executar o agente
        result = agent.run_sync(f"Extraia as referências bibliográficas do seguinte texto de artigo científico:\n\n{limited_text}")
        
        # Converter para lista de dicionários para compatibilidade com DataFrame
        references_list = []
        for ref in result.output.references:
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

def extract_references_with_regex(text):
    """Extrai referências usando expressões regulares em todo o texto"""
    try:
        references = []
        
        # Processar cada padrão
        for pattern_index, pattern in enumerate(REFERENCE_PATTERNS):
            reflist = re.findall(pattern, text, re.MULTILINE | re.UNICODE | re.DOTALL| re.VERBOSE)

            if reflist:
                for ref_match in reflist:
                    groups = ref_match
                    
                    if len(groups) >= 4:
                        authors = groups[0].strip()

                        # Para o padrão numerado especial (6 grupos)
                        if len(groups) == 6:
                            title = groups[1].strip()
                            journal = groups[2].strip()
                            volume = groups[3].strip()
                            pages = groups[4].strip()
                            year = groups[5].strip()
                        # Para o padrão 7 (formato [número] Autor: Título, Editora (ano))
                        elif pattern_index == 7:
                            title = groups[1].strip()
                            journal = groups[2].strip()
                            year = groups[3].strip()
                            volume = ""
                        else:
                            # Para outros padrões (4 grupos)
                            year = groups[1].strip()
                            title = groups[2].strip()
                            journal = groups[3].strip()
                            volume = ""


                        # Extrair DOI se presente
                        doi_match = re.search(r'doi[:\s]*([^\s,]+)', journal, re.IGNORECASE)
                        doi = doi_match.group(1) if doi_match else ""

                        # Extrair volume e páginas (se não foram extraídos pelo padrão especial)
                        if len(groups) != 6:
                            vol_pages_match = re.search(r'(\d+)\s*\(?\d*\)?\s*[,:]\s*(\d+[-–]\d+)', journal)
                            volume = vol_pages_match.group(1) if vol_pages_match else ""
                            pages = vol_pages_match.group(2) if vol_pages_match else ""
                        else:
                            # Para o padrão numerado, extrair páginas do journal
                            pages_match = re.search(r'(\d+[-–]\d+)', journal)
                            pages = pages_match.group(1) if pages_match else ""

                        # Limpar campos
                        authors = re.sub(r'\s+', ' ', authors)
                        title = re.sub(r'\s+', ' ', title)
                        journal = re.sub(r'\s+', ' ', journal)

                        reference = {
                            "authors": authors,
                            "title": title,
                            "journal": journal,
                            "year": year,
                            "volume": volume,
                            "pages": pages,
                            "doi": doi
                        }

                        references.append(reference)


        return references
        
    except Exception as e:
        return [{"error": f"Erro na extração por regex: {str(e)}"}]

def create_plain_text(text, regex_references):
    """Retorna o texto extraído como texto simples"""
    try:
        return text
        
    except Exception as e:
        return f"Erro ao processar texto: {str(e)}"

def process_pdf(pdf_file, model_name):
    """Função principal que processa o PDF e retorna resultados"""
    if pdf_file is None:
        return {"error": "Nenhum arquivo enviado"}, pd.DataFrame(), pd.DataFrame(), "❌ Nenhum arquivo enviado", "Nenhum texto para exibir"
    
    # Extrair texto do PDF
    text, metadata = extract_pdf_text(pdf_file)
    
    if text is None:
        return metadata, pd.DataFrame(), pd.DataFrame(), "❌ Erro ao processar PDF", "Erro ao extrair texto"
    
    # Adicionar modelo selecionado aos metadados
    metadata["modelo_usado"] = model_name
    metadata["caracteres_extraidos"] = len(text)
    metadata["palavras_aproximadas"] = len(text.split())
    
    # Extrair referências com LLM
    llm_references = extract_references_with_llm(text, model_name)
    
    # Extrair referências com Regex
    regex_references = extract_references_with_regex(text)
    
    # Criar texto simples
    plain_text = create_plain_text(text, regex_references)
    
    # Converter para DataFrames
    if llm_references and not any("error" in ref for ref in llm_references):
        llm_df = pd.DataFrame(llm_references)
    else:
        llm_df = pd.DataFrame({"Erro": ["Não foi possível extrair referências com LLM"]})
    
    if regex_references and not any("error" in ref for ref in regex_references):
        regex_df = pd.DataFrame(regex_references)
    else:
        regex_df = pd.DataFrame({"Erro": ["Não foi possível extrair referências com Regex"]})
    
    # Criar status
    llm_count = len(llm_references) if llm_references and not any("error" in ref for ref in llm_references) else 0
    regex_count = len(regex_references) if regex_references and not any("error" in ref for ref in regex_references) else 0
    
    status = f"📊 **Resultados da Extração:**\n- LLM ({model_name}): {llm_count} referências\n- Regex: {regex_count} referências"
    
    return metadata, llm_df, regex_df, status, plain_text

def create_interface():
    """Cria a interface Gradio"""
    with gr.Blocks(title="Extrator de Referências") as interface:
        gr.Markdown("# 📚 Extrator de Referências de Artigos Científicos")
        gr.Markdown("Faça upload de um PDF de artigo científico para extrair automaticamente a lista de referências usando IA e expressões regulares.")
        
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
                extracted_text_output = gr.Textbox(
                    label="📄 Texto Extraído",
                    lines=20,
                    max_lines=20,
                    show_copy_button=True,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                llm_references_output = gr.Dataframe(
                    label="🤖 Referências Extraídas por IA",
                    row_count=(10,'dynamic'),
                    show_copy_button=True,
                    show_fullscreen_button=True,
                    wrap=True
                )
            with gr.Column():
                regex_references_output = gr.Dataframe(
                    label="🔍 Referências Extraídas por Regex",
                    row_count=(10,'dynamic'),
                    show_copy_button=True,
                    show_fullscreen_button=True,
                    wrap=True
                )
        
        status_output = gr.Markdown(label="📊 Status da Extração")
        
        extract_btn.click(
            process_pdf,
            inputs=[pdf_input, model_dropdown],
            outputs=[metadata_output, llm_references_output, regex_references_output, status_output, extracted_text_output]
        )
    
    return interface

def main():
    load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
    
    # Verificar se as chaves das APIs estão configuradas
    google_key = os.getenv("GEMINI_API_KEY")
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
