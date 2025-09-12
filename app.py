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

def extract_references_with_regex(text):
    """Extrai referências usando expressões regulares"""
    try:
        # Encontrar a seção de referências
        references_section = ""
        
        # Padrões para identificar início da seção de referências
        ref_patterns = [
            r'(?i)references?\s*\n',
            r'(?i)bibliography\s*\n',
            r'(?i)literatura\s+citada\s*\n',
            r'(?i)referências\s+bibliográficas\s*\n'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text)
            if match:
                references_section = text[match.end():]
                break
        
        if not references_section:
            # Se não encontrou seção específica, usar últimos 30% do texto
            references_section = text[int(len(text) * 0.7):]
        
        # Padrões para extrair referências individuais
        # Padrão básico: Autor(es). (Ano). Título. Journal/Editora.
        ref_pattern = r'([A-Z][^.]*?)\.\s*\((\d{4})\)\.\s*([^.]+)\.\s*([^.]+?)(?:\.|$)'
        
        # Padrão alternativo para referências numeradas
        numbered_pattern = r'\[\d+\]\s*([A-Z][^.]*?)\.\s*\((\d{4})\)\.\s*([^.]+)\.\s*([^.]+?)(?:\.|$)'
        
        # Padrão para referências com formato diferente
        alt_pattern = r'([A-Z][A-Za-z\s,&]+)\s+\((\d{4})\)[.,]\s*([^.]+)[.,]\s*([^.]+?)(?:\.|$)'
        
        references = []
        
        # Tentar diferentes padrões
        for pattern in [ref_pattern, numbered_pattern, alt_pattern]:
            matches = re.findall(pattern, references_section, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                if len(match) >= 4:
                    # Limpar e processar os dados extraídos
                    authors = match[0].strip()
                    year = match[1].strip()
                    title = match[2].strip()
                    journal = match[3].strip()
                    
                    # Extrair DOI se presente
                    doi_match = re.search(r'doi[:\s]*([^\s]+)', journal, re.IGNORECASE)
                    doi = doi_match.group(1) if doi_match else ""
                    
                    # Extrair volume e páginas
                    vol_pages_match = re.search(r'(\d+)\s*\(?\d*\)?\s*[,:]\s*(\d+[-–]\d+)', journal)
                    volume = vol_pages_match.group(1) if vol_pages_match else ""
                    pages = vol_pages_match.group(2) if vol_pages_match else ""
                    
                    references.append({
                        "authors": authors,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "volume": volume,
                        "pages": pages,
                        "doi": doi
                    })
        
        # Remover duplicatas baseadas no título
        seen_titles = set()
        unique_references = []
        for ref in references:
            title_key = ref["title"].lower().strip()
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_references.append(ref)
        
        return unique_references[:50]  # Limitar a 50 referências para evitar ruído
        
    except Exception as e:
        return [{"error": f"Erro na extração por regex: {str(e)}"}]

def create_highlighted_text(text, regex_references):
    """Cria HTML com texto destacado onde foram encontradas referências por regex"""
    try:
        # Encontrar a seção de referências
        references_section = ""
        section_start = 0
        
        # Padrões para identificar início da seção de referências
        ref_patterns = [
            r'(?i)references?\s*\n',
            r'(?i)bibliography\s*\n',
            r'(?i)literatura\s+citada\s*\n',
            r'(?i)referências\s+bibliográficas\s*\n'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text)
            if match:
                section_start = match.start()
                references_section = text[match.end():]
                break
        
        if not references_section:
            # Se não encontrou seção específica, usar últimos 30% do texto
            section_start = int(len(text) * 0.7)
            references_section = text[section_start:]
        
        # Criar HTML base
        html_text = text.replace('\n', '<br>')
        
        # Cores para diferentes tipos de matches
        colors = ['#ffeb3b', '#4caf50', '#2196f3', '#ff9800', '#9c27b0']
        
        # Padrões para destacar
        patterns = [
            (r'([A-Z][^.]*?)\.\s*\((\d{4})\)\.\s*([^.]+)\.\s*([^.]+?)(?:\.|$)', 'Padrão básico'),
            (r'\[\d+\]\s*([A-Z][^.]*?)\.\s*\((\d{4})\)\.\s*([^.]+)\.\s*([^.]+?)(?:\.|$)', 'Padrão numerado'),
            (r'([A-Z][A-Za-z\s,&]+)\s+\((\d{4})\)[.,]\s*([^.]+)[.,]\s*([^.]+?)(?:\.|$)', 'Padrão alternativo'),
            (r'(?i)references?\s*\n', 'Seção de referências'),
            (r'(?i)bibliography\s*\n', 'Bibliografia')
        ]
        
        # Aplicar destaques
        for i, (pattern, description) in enumerate(patterns):
            color = colors[i % len(colors)]
            
            # Encontrar matches no texto da seção de referências
            section_html = references_section.replace('\n', '<br>')
            matches = list(re.finditer(pattern, references_section, re.MULTILINE | re.DOTALL))
            
            # Destacar matches (processar de trás para frente para não afetar posições)
            for match in reversed(matches):
                start, end = match.span()
                matched_text = references_section[start:end]
                highlighted = f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{description}">{matched_text.replace(chr(10), "<br>")}</span>'
                
                # Calcular posição no texto completo
                full_start = section_start + start
                full_end = section_start + end
                
                # Substituir no HTML completo
                before = html_text[:full_start].replace('\n', '<br>')
                after = html_text[full_end:].replace('\n', '<br>')
                html_text = before + highlighted + after
        
        # Criar HTML final com estilo
        styled_html = f"""
        <div style="
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fafafa;
            white-space: pre-wrap;
        ">
            <div style="margin-bottom: 10px; font-weight: bold; color: #333;">
                📄 Texto Extraído com Destaques das Referências
            </div>
            <div style="margin-bottom: 15px; font-size: 11px; color: #666;">
                <span style="background-color: #ffeb3b; padding: 2px;">■</span> Padrão básico &nbsp;
                <span style="background-color: #4caf50; padding: 2px;">■</span> Padrão numerado &nbsp;
                <span style="background-color: #2196f3; padding: 2px;">■</span> Padrão alternativo &nbsp;
                <span style="background-color: #ff9800; padding: 2px;">■</span> Seção referências
            </div>
            {html_text}
        </div>
        """
        
        return styled_html
        
    except Exception as e:
        return f"<div style='color: red;'>Erro ao criar texto destacado: {str(e)}</div>"

def process_pdf(pdf_file, model_name):
    """Função principal que processa o PDF e retorna resultados"""
    if pdf_file is None:
        return {"error": "Nenhum arquivo enviado"}, pd.DataFrame(), pd.DataFrame(), "❌ Nenhum arquivo enviado", "<div>Nenhum texto para exibir</div>"
    
    # Extrair texto do PDF
    text, metadata = extract_pdf_text(pdf_file)
    
    if text is None:
        return metadata, pd.DataFrame(), pd.DataFrame(), "❌ Erro ao processar PDF", "<div style='color: red;'>Erro ao extrair texto</div>"
    
    # Adicionar modelo selecionado aos metadados
    metadata["modelo_usado"] = model_name
    metadata["caracteres_extraidos"] = len(text)
    metadata["palavras_aproximadas"] = len(text.split())
    
    # Extrair referências com LLM
    llm_references = extract_references_with_llm(text, model_name)
    
    # Extrair referências com Regex
    regex_references = extract_references_with_regex(text)
    
    # Criar HTML com destaques
    highlighted_html = create_highlighted_text(text, regex_references)
    
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
    
    return metadata, llm_df, regex_df, status, highlighted_html

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
                extracted_text_output = gr.HTML(
                    label="📄 Texto Extraído com Destaques",
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
