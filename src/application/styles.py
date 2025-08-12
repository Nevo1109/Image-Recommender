def get_index_string():
    """Returns HTML template string with embedded CSS and JavaScript"""
    return """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%css%}
    <style>
        /* Base Setup */
        html, body {
            height: 100vh !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-size: 14px !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600 !important;
        }
        
        /* Image Styles */
        .img-hover:hover { 
            border-color: #c181fd !important; 
            box-shadow: 0 4px 8px rgba(193, 129, 253, 0.3); 
        }
        .img-responsive { 
            display: block; 
            margin: 0 auto; 
            transition: all 0.3s ease; 
        }
        
        /* Accordion Styles */
        .accordion-sketchy .accordion-item,
        .accordion-clean .accordion-item { 
            background-color: #555555 !important;
            overflow: visible !important;
        }
        .accordion-sketchy .accordion-button,
        .accordion-clean .accordion-button { 
            background-color: #555555 !important; 
            color: #eeccff !important; 
            border: none !important; 
            font-weight: bold; 
            box-shadow: none !important; 
            justify-content: center !important;
            overflow: visible !important;
        }
        .accordion-sketchy .accordion-button:not(.collapsed) { 
            background-color: #eeeeee !important; 
            color: #52009e !important; 
            font-size: bigger;
        }
        .accordion-clean .accordion-button::after { 
            display: none !important;
        }
        .accordion-sketchy .accordion-body { 
            background-color: transparent !important; 
            border: 1px solid #495057 !important; 
            border-radius: 50px !important;
            overflow: visible !important;
        }
        
        /* Progress & Buttons */
        .progress-btn-container { 
            position: relative !important; 
            z-index: 10 !important; 
            pointer-events: auto !important;
            overflow: visible !important;
        }
        
        div[id*="progress-container"] {
            background-color: transparent !important;
        }
        div[id*="progress-container"]:not([style*="display: none"]) {
            background-color: rgba(108, 117, 125, 0.3) !important;
            border: 1px solid rgba(193, 129, 253, 0.3) !important;
        }
        
        button[id*="action-btn"] { 
            pointer-events: auto !important; 
            z-index: 15 !important; 
            position: relative !important; 
        }
        
        /* Custom Transparent Button */
        .custom-transparent-btn {
            background-color: transparent !important;
            background-image: none !important;
            border: 2px solid #c181fd !important;
            border-radius: 6px !important;
            color: #c181fd !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
            width: 100% !important;
            height: 100% !important;
        }
        
        /* Force override Bootstrap button styles */
        .btn.custom-transparent-btn,
        .btn-outline-secondary.custom-transparent-btn,
        button[id*="action-btn"].custom-transparent-btn {
            background-color: transparent !important;
            background: transparent !important;
            background-image: none !important;
            background-clip: border-box !important;
            border: 2px solid #c181fd !important;
            border-radius: 6px !important;
            color: #c181fd !important;
            box-sizing: border-box !important;
        }
        
        .custom-transparent-btn:hover {
            background-color: rgba(193, 129, 253, 0.1) !important;
            border-color: #dab2ff !important;
            color: #dab2ff !important;
            box-shadow: 0 0 15px rgba(193, 129, 253, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        
        .custom-transparent-btn:active,
        .custom-transparent-btn:focus {
            background-color: rgba(193, 129, 253, 0.05) !important;
            border-color: #c181fd !important;
            color: #c181fd !important;
            transform: translateY(0px) !important;
            box-shadow: 0 0 8px rgba(193, 129, 253, 0.4) !important;
            outline: none !important;
        }
        
        /* Filled Add Button */
        .custom-add-btn-filled {
            background-color: #c181fd !important;
            background-image: none !important;
            border: 2px solid #c181fd !important;
            color: white !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(193, 129, 253, 0.3) !important;
        }

        .custom-add-btn-filled:hover {
            background-color: #a855f7 !important;
            border-color: #a855f7 !important;
            box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4) !important;
            transform: translateY(-2px) !important;
        }

        .custom-add-btn-filled:active,
        .custom-add-btn-filled:focus {
            background-color: #9333ea !important;
            border-color: #9333ea !important;
            transform: translateY(-1px) !important;
            outline: none !important;
        }
        
        /* Dropdown Fixes */
        .dropdown-container,
        .dropdown-row,
        .accordion-item,
        .accordion-body {
            overflow: visible !important;
            z-index: auto !important;
        }
        
        .Select-menu-outer {
            z-index: 9999 !important;
            position: absolute !important;
            background: white !important;
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        .Select-option {
            padding: 8px 12px !important;
            background: white !important;
            color: #333 !important;
            cursor: pointer !important;
        }
        
        .Select-option:hover {
            background-color: #f5f5f5 !important;
        }
        
        .Select-option.is-selected {
            background-color: #007bff !important;
            color: white !important;
        }
        
        /* Slider Styling */
        .rc-slider-rail { 
            background-color: #6c757d !important; 
            height: 4px !important; 
        }
        .rc-slider-track { 
            background-color: #c181fd !important; 
            height: 4px !important; 
        }
        .rc-slider-handle { 
            border: 2px solid #c181fd !important;
            background-color: #c181fd !important;
            width: 12px !important; 
            height: 12px !important; 
            margin-top: -4px !important;
            margin-left: 0px !important;
            border-radius: 50% !important;
            cursor: pointer !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(193, 129, 253, 0.3) !important;
            transition: none !important;
        }
        .rc-slider-handle:hover { 
            border-color: #a855f7 !important; 
            background-color: #a855f7 !important;
            box-shadow: 0 0 0 5px rgba(193, 129, 253, 0.4) !important; 
            width: 14px !important;
            height: 14px !important;
            margin-top: -5px !important;
            margin-left: -1px !important;
            transition: all 0.1s ease !important;
        }
        .rc-slider-handle:focus, 
        .rc-slider-handle:active { 
            border-color: #c181fd !important; 
            background-color: #c181fd !important;
            box-shadow: 0 0 0 2px rgba(193, 129, 253, 0.3) !important;
            outline: none !important;
            transition: none !important;
        }
        .rc-slider-mark-text { 
            color: white !important; 
            font-size: 14px !important; 
        }
        .rc-slider-dot { 
            background-color: #6c757d !important;
            border-color: #6c757d !important;
        }
        .rc-slider-dot-active { 
            background-color: #c181fd !important;
            border-color: #c181fd !important;
        }
        
        /* Switch Styling */
        .d-flex.justify-content-center.gap-4.mb-2 {
            display: flex !important;
            justify-content: space-evenly !important;
            gap: 0 !important;
            margin-bottom: 0.5rem !important;
            width: 100% !important;
        }
        
        .switch-container { 
            display: flex !important; 
            align-items: center !important; 
            justify-content: center !important; 
            height: 40px !important;
            flex: 1 !important;
            min-width: 0 !important;
        }
        
        .switch-container .form-check { 
            margin: 0; 
            display: flex; 
            align-items: center !important; /* Vertical centering */
            justify-content: center !important;
            gap: 12px;
            flex-direction: row-reverse; /* Label left, Switch right */
            height: 100% !important;
        }
        
        .switch-container .form-check-label {
            margin: 0 !important;
            line-height: 1.25rem !important;
            display: flex !important;
            align-items: center !important; /* Center label vertically */
        }
        
        .switch-container .form-check-input {
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            appearance: none !important;
            background-image: none !important;
            background-position: none !important;
            background-size: 0 !important;
            background-repeat: no-repeat !important;
            
            width: 2.5rem !important;
            height: 1.25rem !important;
            background-color: #8a919a !important;
            border: 2px solid #8a919a !important;
            border-radius: 1rem !important;
            cursor: pointer !important;
            position: relative !important;
            outline: none !important;
            transition: all 0.2s ease !important;
            margin: 0 !important;
            margin-left: 12px !important;
            box-shadow: none !important;
            overflow: hidden !important;
            flex-shrink: 0 !important; /* Switch should not shrink */
        }
        
        /* Hide Bootstrap's default pseudo-elements */
        .switch-container .form-check-input::after {
            display: none !important;
            content: none !important;
            background: none !important;
        }
        
        /* Custom toggle knob */
        .switch-container .form-check-input::before {
            content: '';
            display: block;
            position: absolute;
            width: 0.9rem;
            height: 0.9rem;
            background-color: white;
            border-radius: 50%;
            top: 50%;
            left: 0.15rem;
            transform: translateY(-50%);
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .switch-container .form-check-input:checked {
            background-color: #c181fd !important;
            border-color: #c181fd !important;
            background-image: none !important;
            background-size: 0 !important;
        }
        
        /* Hide Bootstrap's checked pseudo-elements */
        .switch-container .form-check-input:checked::after {
            display: none !important;
            content: none !important;
            background: none !important;
        }
        
        .switch-container .form-check-input:checked::before {
            left: 1.45rem;
        }
        
        .switch-container .form-check-input:hover {
            transform: scale(1.05);
        }
        .switch-container .form-check-input:not(:checked):hover {
            background-color: #adb5bd !important;
            border-color: #adb5bd !important;
        }
        .switch-container .form-check-input:checked:hover {
            background-color: #a855f7 !important;
            border-color: #a855f7 !important;
        }
        .switch-container .form-check-input:hover::before {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        
        /* Nuclear option - Override Bootstrap switch styles */
        .form-switch .form-check-input {
            background-image: none !important;
            background-position: left center !important;
            background-size: contain !important;
        }
        .form-switch .form-check-input:checked {
            background-image: none !important;
            background-position: right center !important;
        }
        
        .switch-container .form-switch .form-check-input,
        .switch-container .form-check-input.form-switch,
        .switch-container input[type="checkbox"].form-check-input {
            background-image: none !important;
            background-position: none !important;
            background-size: 0 !important;
        }
        
        /* Image Action Buttons Styling - Updated Colors */
        
        /* + Button - Clean White */
        #browse-images-btn,
        button[id="browse-images-btn"],
        .image-action-btn.btn-add {
            background-color: white !important;
            background: white !important;
            border-color: #dee2e6 !important;
            color: #495057 !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            font-size: 14px !important;
            line-height: 1 !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            margin: 0 2px !important;
            border-width: 2px !important;
            background-image: none !important;
            text-shadow: none !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        #browse-images-btn:hover,
        button[id="browse-images-btn"]:hover,
        .image-action-btn.btn-add:hover {
            background-color: #f8f9fa !important;
            background: #f8f9fa !important;
            border-color: #adb5bd !important;
            color: #343a40 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
            transform: translateY(-2px) !important;
            background-image: none !important;
        }
        
        #browse-images-btn:active,
        #browse-images-btn:focus,
        button[id="browse-images-btn"]:active,
        button[id="browse-images-btn"]:focus,
        .image-action-btn.btn-add:active,
        .image-action-btn.btn-add:focus {
            background-color: #e9ecef !important;
            background: #e9ecef !important;
            border-color: #6c757d !important;
            color: #495057 !important;
            transform: translateY(-1px) !important;
            outline: none !important;
            background-image: none !important;
            box-shadow: 0 0 0 0.2rem rgba(108, 117, 125, 0.25) !important;
        }
        
        /* Random Button - Keep Current Purple (unchanged) */
        #random-image-btn,
        button[id="random-image-btn"],
        .image-action-btn.btn-random {
            background-color: #9977CC !important;
            background: #9977CC !important;
            border-color: #9977CC !important;
            color: white !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            font-size: 16px !important;
            line-height: 1 !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            margin: 0 2px !important;
            border-width: 2px !important;
            background-image: none !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
        }
        
        #random-image-btn:hover,
        button[id="random-image-btn"]:hover,
        .image-action-btn.btn-random:hover {
            background-color: #673AB7 !important;
            background: #673AB7 !important;
            border-color: #5E35B1 !important;
            color: white !important;
            box-shadow: 0 0 12px rgba(126, 87, 194, 0.6) !important;
            transform: translateY(-2px) !important;
            background-image: none !important;
        }
        
        #random-image-btn:active,
        #random-image-btn:focus,
        button[id="random-image-btn"]:active,
        button[id="random-image-btn"]:focus,
        .image-action-btn.btn-random:active,
        .image-action-btn.btn-random:focus {
            background-color: #5E35B1 !important;
            background: #5E35B1 !important;
            border-color: #512DA8 !important;
            color: white !important;
            transform: translateY(-1px) !important;
            outline: none !important;
            background-image: none !important;
            box-shadow: 0 0 0 0.2rem rgba(126, 87, 194, 0.25) !important;
        }
        
        /* Delete Button - Rich Red */
        #clear-all-btn,
        button[id="clear-all-btn"],
        .image-action-btn.btn-delete {
            background-color: #dc3545 !important;
            background: #dc3545 !important;
            border-color: #dc3545 !important;
            color: white !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            font-size: 14px !important;
            line-height: 1 !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            margin: 0 2px !important;
            border-width: 2px !important;
            background-image: none !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
        }
        
        #clear-all-btn:hover,
        button[id="clear-all-btn"]:hover,
        .image-action-btn.btn-delete:hover {
            background-color: #c82333 !important;
            background: #c82333 !important;
            border-color: #bd2130 !important;
            color: white !important;
            box-shadow: 0 0 12px rgba(220, 53, 69, 0.6) !important;
            transform: translateY(-2px) !important;
            background-image: none !important;
        }
        
        #clear-all-btn:active,
        #clear-all-btn:focus,
        button[id="clear-all-btn"]:active,
        button[id="clear-all-btn"]:focus,
        .image-action-btn.btn-delete:active,
        .image-action-btn.btn-delete:focus {
            background-color: #bd2130 !important;
            background: #bd2130 !important;
            border-color: #b21f2d !important;
            color: white !important;
            transform: translateY(-1px) !important;
            outline: none !important;
            background-image: none !important;
            box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25) !important;
        }
    </style>
</head>
<body style="margin:0;padding:0;height:100vh;background:#212529;">
    {%app_entry%}
    {%config%}
    {%scripts%}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            function removeFocusAfterInteraction() {
                document.querySelectorAll('.rc-slider-handle, .switch-container .form-check-input, button[id$="-add-btn"], .custom-add-btn-filled, #random-image-btn, .custom-transparent-btn, button[id*="action-btn"], .image-action-btn').forEach(element => {
                    element.addEventListener(element.type === 'checkbox' ? 'change' : 'click', () => {
                        setTimeout(() => element.blur(), 10);
                    });
                });
            }
            
            function fixDropdownContainers() {
                document.querySelectorAll('.dropdown-container, .dropdown-row, .accordion-item, .accordion-body').forEach(container => {
                    container.style.overflow = 'visible';
                    container.style.zIndex = 'auto';
                });
            }
            
            removeFocusAfterInteraction();
            fixDropdownContainers();
            
            const observer = new MutationObserver(() => {
                setTimeout(() => {
                    removeFocusAfterInteraction();
                    fixDropdownContainers();
                }, 50);
            });
            
            observer.observe(document.body, { childList: true, subtree: true });
        });
    </script>
    {%renderer%}
</body>
</html>"""