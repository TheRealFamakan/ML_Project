document.addEventListener('DOMContentLoaded', () => {
    // ---- 0. LOGIN HANDLER & PERSISTENCE ----
    const loginOverlay = document.getElementById('loginOverlay');
    const loginForm = document.getElementById('loginForm');

    // Check Authentication on Load
    if (localStorage.getItem('isAuth') === 'true') {
        loginOverlay.classList.add('hidden');
    }

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const inputs = loginForm.querySelectorAll('input');
        const username = inputs[0].value;
        const password = inputs[1].value;
        
        const btn = loginForm.querySelector('button');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Vérification...';
        btn.disabled = true;

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            
            const data = await response.json();

            if (data.success) {
                // Success
                setTimeout(() => {
                    loginOverlay.classList.add('hidden');
                    localStorage.setItem('isAuth', 'true'); // Persist login
                    
                    // SAVE USER PROFILE BASED ON LOGIN
                    const newProfile = {
                        name: data.user,
                        role: 'Admin / Expert I.A.',
                        avatar: `https://ui-avatars.com/api/?name=${encodeURIComponent(data.user)}&background=random&color=fff`
                    };
                    localStorage.setItem('userProfile', JSON.stringify(newProfile));
                    
                    showToast(`Bienvenue, ${data.user}`, 'success');
                    sessionStorage.setItem('authToken', data.token);

                    // Reload to update UI everywhere
                    setTimeout(() => window.location.reload(), 500);

                }, 800);
            } else {
                // Error
                showToast(data.message || 'Erreur de connexion', 'error');
                btn.innerHTML = originalText;
                btn.disabled = false;
                // Shake animation
                const box = loginForm.closest('.login-box');
                if(box) {
                    box.classList.add('shake');
                    setTimeout(() => box.classList.remove('shake'), 500);
                }
            }
        } catch (error) {
            console.error(error);
            showToast('Erreur serveur', 'error');
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });

    // ---- PRE-INIT: LOAD PROFILE ----
    // Reset legacy profile if needed
    const savedProfile = localStorage.getItem('userProfile');
    if(savedProfile && savedProfile.includes('Dr. Admin')) {
        localStorage.removeItem('userProfile');
    }

    let userProfile = JSON.parse(localStorage.getItem('userProfile')) || {
        name: 'Dr. Camara Famakan',
        role: 'Admin / Lead Data Scientist',
        avatar: 'https://ui-avatars.com/api/?name=CF&background=4f46e5&color=fff'
    };

    // ---- 1. MEMBERS DATA & RENDER ----
    const teamMembers = [
        { name: "Dr. Camara Famakan (Moi)", role: userProfile.role, status: "online", img: userProfile.avatar },
        { name: "Dr. Chamani Fatima", role: "Admin / Expert I.A.", status: "online", img: "https://ui-avatars.com/api/?name=CF&background=ec4899&color=fff" }
    ];

    const membersGrid = document.getElementById('membersGrid');
    
    // Generate Random Contributions for the Chart
    function generateContributions() {
        let html = '<div class="contribution-chart">';
        for(let i=0; i<14; i++) { // 2 weeks of dots
            const level = Math.floor(Math.random() * 4); // 0 to 3
            const cls = level === 0 ? '' : `contrib-c${level}`;
            html += `<div class="contrib-box ${cls}"></div>`;
        }
        html += '</div>';
        return html;
    }

    function renderMembers() {
        if(!membersGrid) return;
        
        // Update count badge
        const badge = document.getElementById('memberCountBadge');
        if(badge) badge.textContent = `${teamMembers.length} en ligne`;

        membersGrid.innerHTML = '';
        teamMembers.forEach(member => {
            const card = document.createElement('div');
            card.className = 'member-card';
            card.innerHTML = `
                <div class="member-avatar-wrapper">
                    <img src="${member.img}" class="member-img">
                    <div class="status-dot status-${member.status}"></div>
                </div>
                <div class="member-name">${member.name}</div>
                <span class="role-badge" onclick="alert('Profil de ${member.name} : ${member.role}')">${member.role}</span>
                ${generateContributions()}
            `;
            membersGrid.appendChild(card);
        });
    }
    renderMembers();
    
    // ---- 2. PROFILE MANAGEMENT ----
    const profileModal = document.getElementById('profileModal');
    const profileForm = document.getElementById('profileForm');
    const userProfileDisplay = document.querySelector('.user-profile');

    function updateProfileUI() {
        // Top Bar
        document.querySelector('.user-info .name').textContent = userProfile.name;
        document.querySelector('.user-info .role').textContent = userProfile.role;
        document.querySelector('.avatar').src = userProfile.avatar;
        
        // Update Members Data
        const myCard = teamMembers.find(m => m.name.includes("Moi"));
        if(myCard) {
            myCard.role = userProfile.role;
            myCard.img = userProfile.avatar;
        }
        renderMembers();
    }
    // Call it once to sync top bar
    document.querySelector('.user-info .name').textContent = userProfile.name;
    document.querySelector('.user-info .role').textContent = userProfile.role;
    document.querySelector('.avatar').src = userProfile.avatar;

    // Make Profile Clickable
    userProfileDisplay.classList.add('clickable-profile');
    userProfileDisplay.addEventListener('click', () => {
        document.getElementById('editNameInput').value = userProfile.name;
        document.getElementById('editRoleInput').value = userProfile.role;
        document.getElementById('editAvatarPreview').src = userProfile.avatar;
        profileModal.classList.remove('hidden');
    });

    document.getElementById('closeProfileBtn').addEventListener('click', () => profileModal.classList.add('hidden'));

    // Avatar Upload Handler
    const avatarInput = document.getElementById('avatarUploadInput');
    let tempAvatar = null;

    if(avatarInput) {
        avatarInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if(file && file.type.match('image.*')) {
                const reader = new FileReader();
                reader.onload = (evt) => {
                    tempAvatar = evt.target.result;
                    document.getElementById('editAvatarPreview').src = tempAvatar;
                };
                reader.readAsDataURL(file);
            }
        });
    }

    profileForm.addEventListener('submit', (e) => {
        e.preventDefault();
        userProfile.name = document.getElementById('editNameInput').value;
        userProfile.role = document.getElementById('editRoleInput').value;
        
        // Use uploaded avatar if exists, else update initials if needed
        if (tempAvatar) {
            userProfile.avatar = tempAvatar;
            tempAvatar = null; // reset
        } else if (userProfile.avatar.includes('ui-avatars.com')) {
            // Only regen initials if not using custom image
            const initials = userProfile.name.split(' ').map(n=>n[0]).join('').substring(0,2);
            userProfile.avatar = `https://ui-avatars.com/api/?name=${initials}&background=0D8ABC&color=fff`;
        }
        
        localStorage.setItem('userProfile', JSON.stringify(userProfile));
        updateProfileUI();
        profileModal.classList.add('hidden');
        showToast('Profil mis à jour', 'success');
    });

    // ---- 3. DASHBOARD LOGIC (Updated for Reset & Multiple) ----
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const folderInput = document.getElementById('folderInput'); // Added folder input
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadedImage = document.getElementById('uploadedImage');
    const previewArea = document.getElementById('imagePreviewArea');
    const uploadContent = document.querySelector('.upload-content');
    const arControls = document.getElementById('arControls');
    const recentList = document.getElementById('recentFilesList');

    let currentFile = null;

    // Enable Multiple Files (Safety)
    if(fileInput) fileInput.setAttribute('multiple', '');

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function handleFiles(files) {
        if (!files.length) return;
        
        let imageFiles = Array.from(files).filter(f => f.type.match('image.*'));
        
        // Extended format check
        if (imageFiles.length === 0) {
             imageFiles = Array.from(files).filter(f => /\.(jpe?g|png|webp|bmp)$/i.test(f.name));
        }

        if (imageFiles.length === 0) {
            showToast('Aucune image valide trouvée', 'error');
            return;
        }

        showToast(`${imageFiles.length} fichiers chargés`, 'success');

        currentBatch = imageFiles; // Save batch
        
        // Update Button if Batch
        if (currentBatch.length > 1) {
            analyzeBtn.innerHTML = `<i class="fa-solid fa-layer-group"></i> Analyser le Lot (${currentBatch.length})`;
            
            // SHOW BATCH GRID PREVIEW
            const grid = document.getElementById('batchPreviewArea');
            grid.innerHTML = ''; // Clear previous
            grid.style.display = 'grid';
            uploadContent.style.display = 'none';
            previewArea.style.display = 'none'; // Hide single view
            analyzeBtn.disabled = false;

            imageFiles.forEach(f => {
                 const reader = new FileReader();
                 reader.onload = (e) => {
                     const div = document.createElement('div');
                     div.className = 'batch-item';
                     div.innerHTML = `<img src="${e.target.result}" title="${f.name}">`;
                     div.onclick = () => loadFileIntoMain(f); // Allow manual preview
                     grid.appendChild(div);
                 };
                 reader.readAsDataURL(f);
             });
             
        } else {
            analyzeBtn.innerHTML = 'Lancer Diagnostic';
            // Single file logic
            // Take the first file for immediate display
            const file = imageFiles[0];
            loadFileIntoMain(file);
        }

        // Add ALL files to sidebar queue (History)
        recentList.innerHTML = '';
        imageFiles.forEach(f => addToSidebarQueue(f));
    }

    function addToSidebarQueue(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'recent-thumb';
            img.title = file.name;
            // Default click behavior (before analysis)
            img.onclick = () => loadFileIntoMain(file);
            recentList.prepend(img);
        };
        reader.readAsDataURL(file);
    }

    let currentBatch = []; // Store multiple files

    function loadFileIntoMain(file) {
        if (!file.type.match('image.*')) {
            showToast('Format non supporté', 'error');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
            previewArea.style.display = 'flex';
            uploadContent.style.display = 'none';
            // Hide Grid if showing single
            document.getElementById('batchPreviewArea').style.display = 'none';
            
            analyzeBtn.disabled = false;
            
            // UI Reset
            document.getElementById('predictionCard').style.display = 'none';
            document.getElementById('batchReportCard').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    dropZone.addEventListener('dragover', () => dropZone.classList.add('drag-over'));
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    
    dropZone.addEventListener('drop', (e) => {
        dropZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    if(folderInput) {
        folderInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
    }

    // Reset Function
    window.resetAnalysis = function() {
        currentFile = null;
        currentBatch = [];
        uploadedImage.src = '';
        analyzeBtn.innerHTML = 'Lancer Diagnostic';
        analyzeBtn.disabled = true;
        previewArea.style.display = 'none';
        uploadContent.style.display = 'flex';
        document.getElementById('predictionCard').style.display = 'none';
        document.getElementById('batchReportCard').style.display = 'none';
        
        // Hide Batch Grid
        const grid = document.getElementById('batchPreviewArea');
        if(grid) {
            grid.style.display = 'none';
            grid.innerHTML = '';
        }
        
        recentList.innerHTML = ''; // Clear batch list
    };

    // ---- 4. ANALYSIS & SKELETON ----
    analyzeBtn.addEventListener('click', async () => {
        if (currentBatch.length > 1) {
            // -- BATCH MODE --
            const formData = new FormData();
            currentBatch.forEach(f => formData.append('files[]', f));
            
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analyse en cours...';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.results) {
                    processBatchResults(data.results);
                }
            } catch (error) {
                console.error(error);
                showToast('Erreur batch', 'error');
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = `<i class="fa-solid fa-check"></i> Lot Analysé`;
            }

        } else {
            // -- SINGLE MODE --
            if (!currentFile) return;

            // UI Loading State
            const loader = document.getElementById('loader');
            const resultsPanel = document.getElementById('predictionCard');
            
            loader.style.display = 'flex';
            resultsPanel.style.display = 'none';
            analyzeBtn.disabled = true;

            const formData = new FormData();
            formData.append('files[]', currentFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.results && data.results.length > 0) {
                    updateResults(data.results[0]);
                }
            } catch (error) {
                console.error(error);
                showToast("Erreur d'analyse", 'error');
            } finally {
                setTimeout(() => {
                    loader.style.display = 'none';
                    analyzeBtn.disabled = false;
                    arControls.style.pointerEvents = 'all';
                }, 1000);
            }
        }
    });

    let lastBatchResults = null; // Store results for export

    function processBatchResults(results) {
        lastBatchResults = results; // Save for export logic

        // Show Card
        const batchCard = document.getElementById('batchReportCard');
        if(batchCard) batchCard.style.display = 'block';

        // Stats
        const total = results.length;
        const pneumonia = results.filter(r => r.prediction === 'PNEUMONIE').length;
        const normal = total - pneumonia;

        // Animate Numbers
        document.getElementById('batchTotal').textContent = total;
        document.getElementById('batchPneumonia').textContent = pneumonia;
        document.getElementById('batchNormal').textContent = normal;

        // Bars
        const pPercent = (pneumonia / total) * 100;
        const nPercent = (normal / total) * 100;
        document.getElementById('batchBarDanger').style.width = `${pPercent}%`;
        document.getElementById('batchBarSuccess').style.width = `${nPercent}%`;

        // UPDATE GRID (Keep Grid in Center)
        const grid = document.getElementById('batchPreviewArea');
        grid.style.display = 'grid';
        grid.innerHTML = '';
        
        document.getElementById('imagePreviewArea').style.display = 'none'; // Ensure Single view is hidden
        document.querySelector('.upload-content').style.display = 'none';

        results.forEach((res, idx) => {
            const div = document.createElement('div');
            div.className = 'batch-item';
            
            // Border Color
            const color = res.prediction === 'PNEUMONIE' ? '#ef4444' : '#22c55e';
            div.style.border = `3px solid ${color}`;
            
            div.innerHTML = `<img src="${res.image_data}" title="${res.filename}">`;
            
            // Click -> Update Right Panel ONLY
            div.onclick = () => {
                // Remove active class from others
                document.querySelectorAll('.batch-item').forEach(b => b.style.transform = 'scale(1)');
                div.style.transform = 'scale(1.05)'; // Highlight
                updateResults(res);
            };
            
            grid.appendChild(div);
        });

        // Show first result in sidebar but keep grid
        updateResults(results[0]);
        showToast(`Lot terminé : ${pneumonia} cas détectés`, 'warning');
    }

    // viewAnalyzedResult Removed (Merged into processBatchResults grid logic)

    let scoreChart = null; // Global var for the chart

    function updateResults(result) {
        const card = document.getElementById('predictionCard');
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreText = document.getElementById('confidenceScore');
        const diagLabel = document.getElementById('diagnosisLabel');
        const metaDate = document.getElementById('metaDate');
        const metaCamera = document.getElementById('metaCamera');
        const metaRes = document.getElementById('metaRes');

        card.style.display = 'block';

        // Update Score Ring
        const color = result.prediction === 'PNEUMONIE' ? '#ef4444' : '#22c55e';
        scoreCircle.style.stroke = color;
        scoreCircle.style.strokeDasharray = `${result.confidence}, 100`;
        scoreText.textContent = `${result.confidence.toFixed(1)}%`;
        scoreText.style.color = color;

        // Update Label
        diagLabel.textContent = result.prediction;
        diagLabel.className = `diagnosis-label ${result.prediction.toLowerCase()}`;
        diagLabel.style.color = color;

        // RENDER CHART
        const ctx = document.getElementById('probaChart').getContext('2d');
        if (scoreChart) scoreChart.destroy();

        scoreChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Normal', 'Pneumonie'],
                datasets: [{
                    label: 'Probabilité (%)',
                    data: [result.probabilities.NORMAL, result.probabilities.PNEUMONIE],
                    backgroundColor: ['#22c55e', '#ef4444'],
                    borderRadius: 5,
                    barThickness: 30
                }]
            },
            options: {
                plugins: { legend: { display: false } },
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100, grid: { display: false } },
                    x: { grid: { display: false } }
                }
            }
        });

        // Update Metadata (Fallback to Real Date if Empty)
        const now = new Date();
        const formattedDate = now.toLocaleDateString('fr-FR', { 
            day: 'numeric', month: 'long', year: 'numeric', 
            hour: '2-digit', minute: '2-digit' 
        });

        if (result.metadata) {
            // If date is Unknown or empty, use current date
            if (!result.metadata.date || result.metadata.date === "Inconnue") {
                metaDate.textContent = formattedDate;
            } else {
                metaDate.textContent = result.metadata.date;
            }
            
            metaCamera.textContent = (result.metadata.camera && result.metadata.camera !== "Inconnue") 
                ? result.metadata.camera 
                : "Capteur X-Ray Standard (Calibration auto)";
                
            metaRes.textContent = result.metadata.resolution || "2048 x 2048 px";
        }
        
        // Add to Timeline
        addToTimeline(result.prediction, formattedDate);

        // Update Daily Stats Helper
        const count = document.getElementById('dailyCount');
        if(count) count.textContent = parseInt(count.textContent || 0) + 1;

        // Draw Heatmap Overlay (Mocked from backend data)
        if (result.heatmap_zones && result.heatmap_zones.length > 0) {
            drawHeatmap(result.heatmap_zones);
        } else {
            clearHeatmap();
        }
    }

    function addToTimeline(diagnosis, date) {
        const timeline = document.getElementById('diagnosticTimeline');
        // Remove empty state if present
        const emptyState = timeline.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const item = document.createElement('div');
        const isRisk = diagnosis === 'PNEUMONIE';
        item.className = `timeline-item ${isRisk ? 'risk' : 'safe'}`;
        
        item.innerHTML = `
            <div class="time-date">${date}</div>
            <div class="time-res">Résultat: ${diagnosis}</div>
        `;
        
        timeline.prepend(item);
    }
    
    // ---- 5. SIDEBAR NAVIGATION & NOTIFS ----
    function setupSidebarNav() {
        // Notifications Logic Removed as requested

        // Nav
        const navLinks = document.querySelectorAll('.nav-menu a');
        
        const actions = {
            'Dashboard': () => document.querySelector('.main-content').scrollTo({ top: 0, behavior: 'smooth' }),
            'Membres': () => document.querySelector('.members-section').scrollIntoView({ behavior: 'smooth' }),
            'Déconnexion': () => {
                localStorage.removeItem('isAuth');
                document.getElementById('loginOverlay').classList.remove('hidden');
                showToast('Déconnexion réussie', 'success');
            }
        };
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                // Active state
                navLinks.forEach(n => n.classList.remove('active'));
                link.classList.add('active');
                
                const text = link.textContent.trim();
                if (actions[text]) actions[text]();
            });
        });

        // Quick Actions (Export PDF)
        const exportBtn = document.querySelector('.btn-quick:nth-child(1)');
        
        if(exportBtn) {
            exportBtn.onclick = async () => {
                showToast('Analyse des données pour le rapport...', 'info');
                
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF();
                
                // Helper: Professional Header
                const addHeader = (titleSuffix) => {
                    doc.setFillColor(13, 138, 188); // Medical Blue
                    doc.rect(0, 0, 210, 40, 'F');
                    
                    doc.setTextColor(255, 255, 255);
                    doc.setFontSize(24);
                    doc.setFont("helvetica", "bold");
                    doc.text("DeepPneumonia", 15, 25);
                    
                    doc.setFontSize(10);
                    doc.setFont("helvetica", "normal");
                    doc.text("Système d'Aide au Diagnostic", 15, 32);
                    
                    doc.setFontSize(14);
                    doc.text(titleSuffix, 195, 25, { align: 'right' });
                    doc.setFontSize(10);
                    doc.text(`Dr. ${userProfile.name} | ID: ${userProfile.id}`, 195, 32, { align: 'right' });
                };

                // Helper: Footer
                const addFooter = () => {
                    const pageCount = doc.internal.getNumberOfPages();
                    for(let i = 1; i <= pageCount; i++) {
                        doc.setPage(i);
                        doc.setFontSize(8);
                        doc.setTextColor(150);
                        doc.text(`Généré le ${new Date().toLocaleString()} - Confidentiel Médical`, 15, 285);
                        doc.text(`Page ${i}/${pageCount}`, 195, 285, { align: 'right' });
                    }
                };

                // DECISION: Single vs Batch
                // We use lastBatchResults which contains the full backend response
                // If it is null or empty, we scrape the DOM for single view.
                
                const isBatchMode = lastBatchResults && lastBatchResults.results && lastBatchResults.results.length > 1;

                if (isBatchMode) {
                    // --- BATCH MODE ---
                    addHeader("LISTE DE CONTRÔLE CLINIQUE");
                    
                    const stats = lastBatchResults.summary;
                    const cleanResults = lastBatchResults.results;

                    // Summary Box
                    doc.setDrawColor(200);
                    doc.setFillColor(245, 247, 250);
                    doc.roundedRect(15, 50, 180, 25, 2, 2, 'F');
                    
                    doc.setFontSize(11);
                    doc.setTextColor(50);
                    doc.text(`Total Dossiers: ${stats.total}`, 25, 66);
                    doc.text(`Cas à Risque: ${stats.pneumonia}`, 85, 66);
                    doc.text(`Cas Normaux: ${stats.normal}`, 145, 66);

                    // Table
                    doc.autoTable({
                        startY: 85,
                        head: [['Ref. Fichier', 'Diagnostic', 'Note Confiance', 'Statut']],
                        body: cleanResults.map(item => [
                            item.filename,
                            item.prediction,
                            item.probability,
                            item.prediction === 'PNEUMONIE' ? 'URGENCE' : 'OK'
                        ]),
                        theme: 'grid',
                        headStyles: { fillColor: [13, 138, 188], textColor: 255 },
                        alternateRowStyles: { fillColor: [240, 240, 240] },
                        didParseCell: function(data) {
                            if (data.section === 'body' && data.column.index === 3) {
                                if (data.cell.raw === 'URGENCE') {
                                    data.cell.styles.textColor = [220, 53, 69]; // Red
                                    data.cell.styles.fontStyle = 'bold';
                                } else {
                                    data.cell.styles.textColor = [40, 167, 69]; // Green
                                }
                            }
                        }
                    });

                    // Add Disclaimer
                    doc.setFontSize(9);
                    doc.setTextColor(100);
                    const finalY = doc.lastAutoTable.finalY || 150;
                    doc.text("Note: Ce rapport récapitulatif identifie les cas prioritaires pour revue manuelle.", 15, finalY + 15);

                } else {
                    // --- SINGLE PATIENT MODE ---
                    addHeader("RAPPORT INDIVIDUEL");

                    // Gather Data from DOM
                    const diagnosis = document.getElementById('diagnosisLabel').textContent;
                    const confidence = document.getElementById('confidenceScore').textContent;
                    
                    // Patient Context Box
                    doc.setDrawColor(180);
                    doc.setFillColor(252, 252, 252);
                    doc.rect(15, 55, 180, 35, 'F');
                    
                    doc.setFontSize(11);
                    doc.setTextColor(80);
                    doc.text("DOSSIER PATIENT", 20, 65);
                    doc.setFontSize(12);
                    doc.setTextColor(0);
                    doc.text(`ID Unique: REF-${Math.floor(Math.random()*100000)}`, 20, 80);
                    doc.text(`Date Examen: ${new Date().toLocaleDateString()}`, 100, 80);

                    // Result Box
                    const isRisk = diagnosis.includes("PNEUMONIE");
                    doc.setFillColor(isRisk ? 255 : 240, isRisk ? 240 : 255, isRisk ? 240 : 250); // Red tinted or Green tinted bg
                    doc.setDrawColor(isRisk ? 200 : 150, 0, 0);
                    doc.roundedRect(15, 105, 180, 60, 2, 2, 'FD');

                    doc.setFontSize(14);
                    doc.setTextColor(50);
                    doc.text("CONCLUSION I.A.", 25, 120);

                    doc.setFontSize(22);
                    doc.setFont("helvetica", "bold");
                    doc.setTextColor(isRisk ? 220 : 40, isRisk ? 20 : 167, isRisk ? 60 : 69);
                    doc.text(diagnosis.toUpperCase(), 25, 135);

                    doc.setFontSize(12);
                    doc.setFont("helvetica", "normal");
                    doc.setTextColor(80);
                    doc.text(`Indice de fiabilité du modèle: ${confidence}`, 25, 150);

                    // Recommendations
                    doc.setFontSize(12);
                    doc.setTextColor(0);
                    doc.text("Recommandations suggérées:", 15, 185);
                    
                    const recs = isRisk 
                        ? ["1. Confirmation radiologique requise.", "2. Bilan sanguin inflammatoire.", "3. Mise en isolement préventif."]
                        : ["1. Aucune anomalie détectée.", "2. Suivi standard.", "3. Rassurer le patient."];
                    
                    let yPos = 200;
                    recs.forEach(rec => {
                        doc.text(rec, 20, yPos);
                        yPos += 10;
                    });
                }
                
                addFooter();
                doc.save(`DeepPneumonia_Report_${Date.now()}.pdf`);
                showToast('Rapport PDF généré avec succès !', 'success');
            };
        }
    }
    setupSidebarNav();

    // ---- 6. MAGNIFIER & OVERLAY ----
    const glass = document.getElementById('magnifierGlass');
    const img = document.getElementById('uploadedImage');
    const zoomLevel = 3;

    function moveMagnifier(e) {
        if (previewArea.style.display === 'none') return;
        
        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if cursor within image
        if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
            glass.style.display = 'none';
            return;
        }

        glass.style.display = 'block';
        glass.style.left = `${x - glass.offsetWidth / 2}px`;
        glass.style.top = `${y - glass.offsetHeight / 2}px`;
        
        // Background Image Position
        glass.style.backgroundImage = `url('${img.src}')`;
        glass.style.backgroundSize = `${rect.width * zoomLevel}px ${rect.height * zoomLevel}px`;
        glass.style.backgroundPosition = `-${x * zoomLevel - glass.offsetWidth / 2}px -${y * zoomLevel - glass.offsetHeight / 2}px`;
    }

    previewArea.addEventListener('mousemove', moveMagnifier);
    previewArea.addEventListener('mouseleave', () => glass.style.display = 'none');

    // Overlay Logic
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const toggle = document.getElementById('overlayToggle');
    const opacitySlider = document.getElementById('opacityRange');

    function drawHeatmap(zones) {
        canvas.width = img.width || 500; 
        canvas.height = img.height || 500;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        zones.forEach(zone => {
            ctx.beginPath();
            ctx.arc(zone.x, zone.y, zone.r, 0, 2 * Math.PI);
            ctx.fillStyle = `rgba(255, 0, 0, ${zone.intensity})`;
            ctx.fill();
        });
        updateOverlayOpacity();
    }

    function clearHeatmap() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function updateOverlayOpacity() {
        canvas.style.opacity = toggle.checked ? opacitySlider.value : '0';
    }

    toggle.addEventListener('change', updateOverlayOpacity);
    opacitySlider.addEventListener('input', updateOverlayOpacity);

    // ---- 7. UTILS ----
    function showToast(msg, type = 'success') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<i class="fa-solid fa-${type === 'success' ? 'check' : 'circle-exclamation'}"></i><span>${msg}</span>`;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
});
