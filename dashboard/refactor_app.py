import os

def refactor_app():
    file_path = "app.py"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. Remove manual auto-refresh mechanism (lines 21-29 roughly)
        if "# Auto-refresh mechanism for live updates" in line:
            # Skip the next 8 lines
            while i < len(lines) and "st.rerun()" not in lines[i]:
                i += 1
            i += 1 # skip st.rerun() line
            # Skip the blank line if any
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
            
        # 2. Replace singletons with st.cache_resource (lines 394-402 roughly)
        if "if '_METRICS_TRACKER_SINGLETON' not in globals():" in line:
            # Skip until ALERTS_MANAGER is set
            while i < len(lines) and "ALERTS_MANAGER = globals()['_ALERTS_MANAGER_SINGLETON']" not in lines[i]:
                i += 1
            i += 1 # skip the last line of that block
            
            new_lines.extend([
                "@st.cache_resource\n",
                "def get_metrics_tracker():\n",
                "    return MetricsTracker()\n",
                "\n",
                "@st.cache_resource\n",
                "def get_alerts_manager():\n",
                "    return AlertsManager()\n",
                "\n",
                "METRICS_TRACKER = get_metrics_tracker()\n",
                "ALERTS_MANAGER = get_alerts_manager()\n"
            ])
            continue
            
        # 3. Replace background thread startup (lines 481-488 roughly)
        if "if '_DASH_METRICS_THREAD_STARTED' not in globals():" in line:
            while i < len(lines) and "thread.start()" not in lines[i]:
                i += 1
            i += 1
            
            new_lines.extend([
                "@st.cache_resource\n",
                "def start_background_thread():\n",
                "    fetch_metrics_once()\n",
                "    thread = threading.Thread(target=update_metrics_thread, daemon=True)\n",
                "    thread.start()\n",
                "    return True\n",
                "\n",
                "start_background_thread()\n"
            ])
            continue
            
        # 4. Wrap everything after navigation pills in a fragment
        if 'selected = st.pills("Navigation", menu_options, default="Dashboard", label_visibility="collapsed")' in line:
            new_lines.append(line)
            new_lines.append("\n")
            
            # Using try/except in case Streamlit version doesn't support fragment, we fallback to just a normal function
            new_lines.extend([
                "try:\n",
                "    fragment_decorator = st.fragment(run_every=2)\n",
                "except AttributeError:\n",
                "    # Fallback if st.fragment is not available (older streamlit)\n",
                "    def fragment_decorator(func):\n",
                "        return func\n",
                "    from streamlit_autorefresh import st_autorefresh\n",
                "    st_autorefresh(interval=2000, key=\"data_refresh\")\n",
                "\n",
                "@fragment_decorator\n",
                "def render_page_content(selected):\n"
            ])
            i += 1
            
            # Indent the rest of the file
            while i < len(lines):
                # Don't indent blank lines blindly
                if lines[i].strip() == "":
                    new_lines.append("\n")
                else:
                    new_lines.append("    " + lines[i])
                i += 1
                
            # Add the function call at the end
            new_lines.extend([
                "\n",
                "render_page_content(selected)\n"
            ])
            break

        new_lines.append(line)
        i += 1

    # Rename original to app_backup.py
    if os.path.exists("app_backup.py"):
        os.remove("app_backup.py")
    os.rename(file_path, "app_backup.py")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Refactoring complete.")

if __name__ == "__main__":
    refactor_app()
