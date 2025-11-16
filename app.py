from system import QnASystem

def some_good_looking_prints():
    print("\n" * 3)
    print("===================================")
    print("      🎯 Welcome to AmbedkarGPT - RAG-based Q&A System     ")
    print("📌 This system answers questions about Dr. B.R. Ambedkar's speech")
    print("   - Type your question and press Enter")
    print("   - For exit type 'quit', 'exit', or 'q'")
    print("===================================")

app = QnASystem("speech.txt", chunk_size=50, chunk_overlap=10)
try:
    app.load_document()
    app.split_document_texts()
    app.create_embeddings()
    app.create_vector_store()
    app.create_qa_chain()

    some_good_looking_prints()

    while True:
        try:
            user_input = input("\n❓ Ask a question: ").strip()
    
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using AmbedkarGPT. Goodbye!")
                break

            app.answer_question(question=user_input)
        except KeyboardInterrupt:
            print("\n\n👋 Application interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")

except Exception as e:
    print(f"Initialization failed: {e}")