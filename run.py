from app import create_app

app = create_app()

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except SystemExit as e:
        print(f"Ocorreu uma exceção SystemExit: {e}")

