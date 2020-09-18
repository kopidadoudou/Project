mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"1161104595@student.mmu.edu.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml