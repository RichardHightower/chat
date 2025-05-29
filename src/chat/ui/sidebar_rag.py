"""
RAG sidebar components for the chat application.

This module contains functions for rendering the RAG-specific sidebar components
including project management, file upload, and RAG settings.
"""

import streamlit as st
import os
from typing import Dict, List, Optional, Tuple

from chat.rag.rag_service import RAGService
from chat.util.logging_util import logger as llm_logger


def initialize_rag_state():
    """Initialize RAG-related session state variables if they don't exist."""
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    if "rag_service" not in st.session_state:
        try:
            from chat.rag.rag_service import RAGService
            st.session_state.rag_service = RAGService()
        except Exception as e:
            st.error(f"Error initializing RAG service: {str(e)}")
            st.session_state.rag_service = None
            st.session_state.rag_enabled = False
            return

    if "current_rag_project_id" not in st.session_state:
        st.session_state.current_rag_project_id = None

    if "rag_similarity_threshold" not in st.session_state:
        st.session_state.rag_similarity_threshold = 0.7

    if "rag_max_chunks" not in st.session_state:
        st.session_state.rag_max_chunks = 3


def render_rag_sidebar():
    """Render the RAG sidebar components."""
    st.header("RAG Settings")

    # Initialize RAG session state
    initialize_rag_state()

    # RAG toggle
    st.session_state.rag_enabled = st.checkbox(
        "Enable RAG",
        value=st.session_state.rag_enabled,
        help="When enabled, the chat will use Retrieval Augmented Generation to provide context from your documents."
    )

    if not st.session_state.rag_enabled:
        st.info("Enable RAG to access document management features.")
        return

    # Project management
    render_project_management()

    # Only show file management and settings if a project is selected
    if st.session_state.current_rag_project_id:
        with st.expander("File Management", expanded=True):
            render_file_management(st.session_state.current_rag_project_id)

        with st.expander("RAG Settings", expanded=True):
            render_rag_settings()


def render_project_management():
    """Render project management components."""
    st.subheader("Project Management")

    # Get service instance
    rag_service = st.session_state.rag_service

    # Get list of projects
    projects = rag_service.list_projects()
    
    # If we have a project ID but no project object, try to restore it
    if (st.session_state.get("current_rag_project_id") and 
        not st.session_state.get("rag_project") and 
        projects):
        project_id = st.session_state.current_rag_project_id
        selected_project = next((p for p in projects if p.id == project_id), None)
        if selected_project:
            st.session_state.rag_project = selected_project

    if projects:
        # Create a dropdown for project selection
        project_options = ["Select a project..."] + [f"{p.name} (ID: {p.id})" for p in projects]
        selected_option = st.selectbox("Select Project", project_options)

        if selected_option != "Select a project...":
            # Extract project ID from selection
            project_id = int(selected_option.split("ID: ")[1].rstrip(")"))
            st.session_state.current_rag_project_id = project_id

            # Display project info
            selected_project = next((p for p in projects if p.id == project_id), None)
            if selected_project:
                # Store both the project ID and the project object
                st.session_state.rag_project = selected_project
                st.success(f"Using project: {selected_project.name}")
                if selected_project.description:
                    st.info(selected_project.description)

    # Create new project section
    st.markdown("---")
    st.markdown("### Create New Project")

    new_project_name = st.text_input("Project Name", key="new_project_name")
    new_project_desc = st.text_area("Project Description", key="new_project_desc")

    if st.button("Create Project"):
        if new_project_name:
            try:
                new_project = rag_service.get_or_create_project(
                    name=new_project_name,
                    description=new_project_desc
                )
                st.session_state.current_rag_project_id = new_project.id
                st.session_state.rag_project = new_project
                st.success(f"Created project: {new_project.name} (ID: {new_project.id})")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating project: {str(e)}")
        else:
            st.warning("Please enter a project name.")


def render_file_management(project_id: int):
    """Render file management components for a project."""
    rag_service = st.session_state.rag_service

    # File upload section
    st.markdown("### Upload Document")

    # Create columns for upload components
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "md", "py", "js", "html", "css", "json", "csv"],
            key=f"file_uploader_{project_id}"
        )

    with col2:
        # Only show the button when a file is selected
        if uploaded_file is not None:
            if st.button("Process File", key=f"process_file_{project_id}"):
                try:
                    # Read file content
                    content = uploaded_file.getvalue().decode("utf-8")

                    # Create metadata
                    metadata = {
                        "filename": uploaded_file.name,
                        "type": uploaded_file.type,
                        "size": uploaded_file.size
                    }

                    # Add file to project
                    file = rag_service.add_document(
                        project_id=project_id,
                        filename=uploaded_file.name,
                        content=content,
                        metadata=metadata
                    )

                    if file:
                        st.success(f"Added file: {file.name}")
                        st.rerun()
                    else:
                        st.error("Failed to add document.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    # List existing files
    st.markdown("### Project Files")

    try:
        files = rag_service.list_files(project_id)

        if not files:
            st.info("No files in this project.")
            return

        for file in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{file.name}")
            with col2:
                if st.button("Remove", key=f"remove_{file.id}_{project_id}"):
                    try:
                        if rag_service.handler.delete_file(file.id):
                            st.success(f"Removed file: {file.name}")
                            st.rerun()
                        else:
                            st.error(f"Failed to remove file: {file.name}")
                    except Exception as e:
                        st.error(f"Error removing file: {str(e)}")
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")


def render_rag_settings():
    """Render RAG settings components."""
    # Similarity threshold
    st.session_state.rag_similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.rag_similarity_threshold,
        step=0.05,
        help="Minimum similarity score for a document chunk to be included in context."
    )

    # Max chunks
    st.session_state.rag_max_chunks = st.slider(
        "Max Chunks",
        min_value=1,
        max_value=10,
        value=st.session_state.rag_max_chunks,
        step=1,
        help="Maximum number of document chunks to include in context."
    )