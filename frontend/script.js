/**
 * Fake Account Detector AI - Frontend Application Controller
 * Handles REST API communication, live health check, input validation,
 * quick test presets, animated gauge rendering, and explainability dashboard.
 * Updated for the simplified 8-feature schema.
 */

// Determine API Base URL dynamically
const getApiBase = () => {
  const origin = window.location.origin;
  if (origin.includes(":5000")) {
    return "";
  }
  return "http://127.0.0.1:5000";
};

const API_BASE = getApiBase();

// Preset test account profiles for hackathon presentation
const TEST_PRESETS = {
  genuine_user: {
    followers_count: 520,
    following_count: 410,
    posts_count: 145,
    account_age_bracket: "3", // 1–2 years
    has_profile_pic: "1",
    has_bio: "1",
    has_external_link: "0",
    account_type: "public"
  },
  spam_bot: {
    followers_count: 14,
    following_count: 3800,
    posts_count: 1,
    account_age_bracket: "0", // Less than 1 month
    has_profile_pic: "0",
    has_bio: "0",
    has_external_link: "0",
    account_type: "public"
  },
  scam_promo: {
    followers_count: 420,
    following_count: 1200,
    posts_count: 3,
    account_age_bracket: "0", // Less than 1 month
    has_profile_pic: "1",
    has_bio: "1",
    has_external_link: "1",
    account_type: "public"
  },
  lurker_bot: {
    followers_count: 2,
    following_count: 5,
    posts_count: 0,
    account_age_bracket: "1", // 1–6 months
    has_profile_pic: "0",
    has_bio: "0",
    has_external_link: "0",
    account_type: "public"
  }
};

const AGE_BRACKET_TEXT = {
  0: "Less than 1 month",
  1: "1–6 months",
  2: "6–12 months",
  3: "1–2 years",
  4: "More than 2 years"
};

// Benchmarks for comparative table
const FEATURE_BENCHMARKS = {
  followers_count: { name: "Followers Count", benchmark: "50 - 5,000+", unit: "users" },
  following_count: { name: "Following Count", benchmark: "50 - 1,500", unit: "users" },
  posts_count: { name: "Number of Posts", benchmark: "20 - 500+", unit: "posts" },
  account_age_bracket: { name: "Account Age", benchmark: "6+ months (Established)", unit: "" },
  has_profile_pic: { name: "Profile Picture", benchmark: "Yes (Custom photo)", unit: "" },
  has_bio: { name: "Has Biography", benchmark: "Yes (Populated)", unit: "" },
  has_external_link: { name: "Bio External Link", benchmark: "No / Verified site", unit: "" },
  is_private: { name: "Account Type", benchmark: "Public or Private", unit: "" }
};

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initPresets();
  initFormHandler();
  initUIInteractions();
});

/**
 * Checks Flask backend health on startup and populates live metrics
 */
async function initHealthCheck() {
  const statusPill = document.getElementById("backend-status-pill");
  const statusText = document.getElementById("backend-status-text");
  const heroAcc = document.getElementById("hero-model-acc");

  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.status === "healthy" && data.model_loaded) {
      statusPill.className = "status-pill status-online";
      statusText.textContent = "AI Engine Online (Flask + Random Forest)";
      fetchMetrics(heroAcc);
    } else {
      statusPill.className = "status-pill status-offline";
      statusText.textContent = "AI Model Not Initialized";
    }
  } catch (err) {
    statusPill.className = "status-pill status-offline";
    statusText.textContent = "Backend Offline (Run: python backend/app.py)";
  }
}

/**
 * Loads model metrics from /api/metrics
 */
async function fetchMetrics(heroAccElem) {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`, { method: "GET" });
    if (!res.ok) return;
    const json = await res.json();
    if (json.success && json.metrics) {
      const acc = json.metrics.accuracy ? (json.metrics.accuracy * 100).toFixed(1) : "100.0";
      if (heroAccElem) {
        heroAccElem.textContent = `${acc}%`;
      }
    }
  } catch (e) {
    // Fallback
  }
}

/**
 * Initializes quick-load preset buttons
 */
function initPresets() {
  const presetButtons = document.querySelectorAll(".preset-btn");
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      const values = TEST_PRESETS[presetKey];
      if (!values) return;

      document.getElementById("followers_count").value = values.followers_count;
      document.getElementById("following_count").value = values.following_count;
      document.getElementById("posts_count").value = values.posts_count;
      
      const ageSelect = document.getElementById("account_age_bracket");
      if (ageSelect) {
        ageSelect.value = values.account_age_bracket;
      }

      setRadioValue("has_profile_pic", values.has_profile_pic);
      setRadioValue("has_bio", values.has_bio);
      setRadioValue("has_external_link", values.has_external_link);
      setRadioValue("account_type", values.account_type || "public");

      clearFormErrors();

      presetButtons.forEach(b => b.classList.remove("active-preset"));
      btn.classList.add("active-preset");
    });
  });
}

function setRadioValue(name, val) {
  const radios = document.querySelectorAll(`input[name="${name}"]`);
  radios.forEach(r => {
    r.checked = (r.value === String(val));
  });
}

function getRadioValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : null;
}

/**
 * Form Submission & Validation Handler
 */
function initFormHandler() {
  const form = document.getElementById("account-form");
  const btnReset = document.getElementById("btn-reset");

  btnReset.addEventListener("click", () => {
    form.reset();
    clearFormErrors();
    const resultsSec = document.getElementById("results-section");
    resultsSec.classList.add("hidden");
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active-preset"));
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormErrors();

    const { isValid, payload, errors } = validateForm();
    if (!isValid) {
      showFormErrors(errors);
      return;
    }

    await submitAnalysis(payload);
  });
}

/**
 * Client-side validation for simplified form
 */
function validateForm() {
  const errors = {};
  const payload = {};

  const numFields = [
    { id: "followers_count", name: "Followers count", min: 0, max: 100000000 },
    { id: "following_count", name: "Following count", min: 0, max: 10000000 },
    { id: "posts_count", name: "Number of posts", min: 0, max: 1000000 }
  ];

  numFields.forEach(f => {
    const el = document.getElementById(f.id);
    const val = el ? el.value.trim() : "";

    if (val === "") {
      errors[f.id] = `${f.name} is required.`;
    } else {
      const num = Number(val);
      if (isNaN(num)) {
        errors[f.id] = `Must be a valid integer.`;
      } else if (num < f.min) {
        errors[f.id] = `Cannot be less than ${f.min}.`;
      } else if (num > f.max) {
        errors[f.id] = `Cannot exceed ${f.max}.`;
      } else {
        payload[f.id] = Math.floor(num);
      }
    }
  });

  // Account Age dropdown
  const ageEl = document.getElementById("account_age_bracket");
  if (!ageEl || ageEl.value === "" || ageEl.value === null) {
    errors["account_age_bracket"] = "Please select an Account Age option.";
  } else {
    payload.account_age_bracket = parseInt(ageEl.value, 10);
  }

  // Segmented controls
  const profilePicVal = getRadioValue("has_profile_pic");
  if (profilePicVal === null) {
    errors["has_profile_pic"] = "Please indicate if a profile picture is present.";
  } else {
    payload.has_profile_pic = parseInt(profilePicVal, 10);
  }

  const bioVal = getRadioValue("has_bio");
  if (bioVal === null) {
    errors["has_bio"] = "Please indicate if the account has a bio.";
  } else {
    payload.has_bio = parseInt(bioVal, 10);
  }

  const linkVal = getRadioValue("has_external_link");
  if (linkVal === null) {
    errors["has_external_link"] = "Please indicate if the bio contains an external link.";
  } else {
    payload.has_external_link = parseInt(linkVal, 10);
  }

  // Account type (Optional, defaults to 'public')
  const accountTypeVal = getRadioValue("account_type") || "public";
  payload.account_type = accountTypeVal;
  payload.is_private = (accountTypeVal === "private") ? 1 : 0;

  return {
    isValid: Object.keys(errors).length === 0,
    payload,
    errors
  };
}

function showFormErrors(errors) {
  const banner = document.getElementById("form-error-banner");
  const msgContainer = document.getElementById("form-error-message");

  let firstField = null;

  for (const [fieldId, msg] of Object.entries(errors)) {
    const inputEl = document.getElementById(fieldId);
    const errText = document.getElementById(`err-${fieldId}`);
    if (inputEl) inputEl.classList.add("input-error");
    if (errText) errText.textContent = msg;

    if (!firstField && inputEl) firstField = inputEl;
  }

  msgContainer.textContent = "Please complete all required fields properly before analyzing.";
  banner.classList.remove("hidden");

  if (firstField) {
    firstField.focus();
  }
}

function clearFormErrors() {
  const banner = document.getElementById("form-error-banner");
  if (banner) banner.classList.add("hidden");

  document.querySelectorAll(".input-error").forEach(el => el.classList.remove("input-error"));
  document.querySelectorAll(".error-text").forEach(el => el.textContent = "");
}

/**
 * Dispatches prediction request to Flask /api/predict
 */
async function submitAnalysis(payload) {
  const btnSubmit = document.getElementById("btn-submit");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnText = btnSubmit.querySelector(".btn-text");
  const resultsSec = document.getElementById("results-section");

  btnSubmit.disabled = true;
  btnSpinner.classList.remove("hidden");
  btnText.style.opacity = "0.4";

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      const errMsg = data.error || (data.details ? data.details.join(" ") : "Analysis failed");
      throw new Error(errMsg);
    }

    renderPredictionResults(data);

    resultsSec.classList.remove("hidden");
    resultsSec.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    const banner = document.getElementById("form-error-banner");
    const msgContainer = document.getElementById("form-error-message");
    msgContainer.textContent = `Server Error: ${err.message}. Ensure backend is active on http://127.0.0.1:5000.`;
    banner.classList.remove("hidden");
  } finally {
    btnSubmit.disabled = false;
    btnSpinner.classList.add("hidden");
    btnText.style.opacity = "1";
  }
}

/**
 * Renders the prediction dashboard
 */
function renderPredictionResults(data) {
  const riskScore = data.risk_score;
  const isSuspicious = (data.prediction === "Suspicious");
  const circumference = 314.159;

  const now = new Date();
  document.getElementById("res-timestamp").textContent = `Analyzed: ${now.toLocaleTimeString()}`;

  const progressCircle = document.getElementById("gauge-progress-circle");
  const riskNumber = document.getElementById("res-risk-percentage");
  const levelBadge = document.getElementById("res-risk-level-badge");

  const offset = circumference * (1 - (riskScore / 100));
  progressCircle.style.strokeDashoffset = offset;

  let strokeColor = "#10b981";
  let levelClass = "status-online";
  if (riskScore >= 75) {
    strokeColor = "#ef4444";
    levelClass = "status-offline";
  } else if (riskScore >= 50) {
    strokeColor = "#f97316";
    levelClass = "status-offline";
  } else if (riskScore >= 25) {
    strokeColor = "#f59e0b";
    levelClass = "status-connecting";
  }

  progressCircle.style.stroke = strokeColor;
  levelBadge.className = `gauge-level-badge ${levelClass}`;
  levelBadge.textContent = data.risk_level;

  animateNumber(riskNumber, 0, riskScore, 1000);

  const pill = document.getElementById("res-classification-pill");
  const heading = document.getElementById("res-verdict-heading");
  const summary = document.getElementById("res-summary-text");
  const confBadge = document.getElementById("res-confidence-badge");

  if (isSuspicious) {
    pill.textContent = "SUSPICIOUS ACCOUNT";
    pill.style.background = "rgba(239, 68, 68, 0.2)";
    pill.style.color = "#f87171";
    pill.style.border = "1px solid rgba(239, 68, 68, 0.4)";
    heading.textContent = "High Probability of Inauthentic Activity";
    heading.style.color = "#fca5a5";
  } else {
    pill.textContent = "LIKELY GENUINE";
    pill.style.background = "rgba(16, 185, 129, 0.2)";
    pill.style.color = "#34d399";
    pill.style.border = "1px solid rgba(16, 185, 129, 0.4)";
    heading.textContent = "Profile Appears Organically Genuine";
    heading.style.color = "#a7f3d0";
  }

  const confidence = isSuspicious ? data.fake_probability : data.genuine_probability;
  confBadge.textContent = `Model Confidence: ${confidence}%`;
  summary.textContent = data.explanation.summary;

  document.getElementById("res-genuine-pct").textContent = `${data.genuine_probability}%`;
  document.getElementById("res-fake-pct").textContent = `${data.fake_probability}%`;
  document.getElementById("prob-fill-genuine").style.width = `${data.genuine_probability}%`;
  document.getElementById("prob-fill-fake").style.width = `${data.fake_probability}%`;

  renderFactorsList(data.explanation.risk_factors, data.explanation.trust_factors);
  renderSummaryTable(data.submitted_features);
}

function animateNumber(element, start, end, duration) {
  let startTime = null;
  const step = (timestamp) => {
    if (!startTime) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);
    const current = (start + (end - start) * progress).toFixed(1);
    element.textContent = `${current}%`;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    } else {
      element.textContent = `${end}%`;
    }
  };
  window.requestAnimationFrame(step);
}

function renderFactorsList(riskFactors, trustFactors) {
  const riskList = document.getElementById("risk-factors-list");
  const trustList = document.getElementById("trust-factors-list");
  const riskCount = document.getElementById("res-risk-count");
  const trustCount = document.getElementById("res-trust-count");

  riskCount.textContent = riskFactors ? riskFactors.length : 0;
  trustCount.textContent = trustFactors ? trustFactors.length : 0;

  if (!riskFactors || riskFactors.length === 0) {
    riskList.innerHTML = `<div class="factor-empty-text">No significant behavioral risk flags triggered.</div>`;
  } else {
    riskList.innerHTML = riskFactors.map(f => `
      <div class="factor-item">
        <div class="factor-item-header">
          <span class="factor-name">${escapeHtml(f.factor)}</span>
          <span class="severity-pill severity-${f.severity || 'medium'}">${escapeHtml(f.severity || 'risk')}</span>
        </div>
        <p class="factor-detail">${escapeHtml(f.detail)}</p>
      </div>
    `).join("");
  }

  if (!trustFactors || trustFactors.length === 0) {
    trustList.innerHTML = `<div class="factor-empty-text">No distinct positive trust signals identified.</div>`;
  } else {
    trustList.innerHTML = trustFactors.map(f => `
      <div class="factor-item">
        <div class="factor-item-header">
          <span class="factor-name text-success">${escapeHtml(f.factor)}</span>
          <span class="severity-pill severity-low">PASS</span>
        </div>
        <p class="factor-detail">${escapeHtml(f.detail)}</p>
      </div>
    `).join("");
  }
}

/**
 * Populates submitted parameters table for the simplified schema
 */
function renderSummaryTable(features) {
  const tbody = document.getElementById("summary-table-body");
  if (!tbody || !features) return;

  const rows = Object.keys(FEATURE_BENCHMARKS).map(key => {
    const meta = FEATURE_BENCHMARKS[key];
    const val = features[key];
    let displayVal = val;
    let evalText = "Within Normal Variance";
    let evalColor = "var(--text-secondary)";

    if (key === "account_age_bracket") {
      displayVal = AGE_BRACKET_TEXT[val] || features.account_age_label || "Selected Option";
      if (val === 0) {
        evalText = "Very Recent (<1 mo)";
        evalColor = "var(--color-danger)";
      } else if (val >= 2) {
        evalText = "Established Account";
        evalColor = "var(--color-success)";
      } else {
        evalText = "Recent Account";
        evalColor = "var(--color-warning)";
      }
    } else if (key === "has_profile_pic") {
      displayVal = val === 1 ? "Yes (Custom Photo)" : "No (Default Avatar)";
      evalText = val === 1 ? "Positive Signal" : "High Risk Flag";
      evalColor = val === 1 ? "var(--color-success)" : "var(--color-danger)";
    } else if (key === "has_bio") {
      displayVal = val === 1 ? "Yes (Populated)" : "No (Empty)";
      evalText = val === 1 ? "Normal" : "Common in Bots";
      evalColor = val === 1 ? "var(--color-success)" : "var(--color-warning)";
    } else if (key === "has_external_link") {
      displayVal = val === 1 ? "Yes (Contains Link)" : "No (No Links)";
      evalText = val === 1 ? "External Link Present" : "Standard Profile";
      evalColor = "var(--text-secondary)";
    } else if (key === "is_private") {
      displayVal = val === 1 ? "Private Profile" : "Public Profile";
      evalText = val === 1 ? "Private (Human Pattern)" : "Public Access";
      evalColor = val === 1 ? "var(--color-success)" : "var(--text-secondary)";
    } else {
      displayVal = `${Number(val).toLocaleString()} ${meta.unit}`;
    }

    return `
      <tr>
        <td><strong>${meta.name}</strong></td>
        <td><code>${displayVal}</code></td>
        <td>${meta.benchmark}</td>
        <td style="color: ${evalColor}; font-weight: 600;">${evalText}</td>
      </tr>
    `;
  });

  tbody.innerHTML = rows.join("");
}

function initUIInteractions() {
  const btnAnalyzeAnother = document.getElementById("btn-analyze-another");
  const btnScrollTop = document.getElementById("btn-scroll-top");

  if (btnAnalyzeAnother) {
    btnAnalyzeAnother.addEventListener("click", () => {
      document.getElementById("account-form").scrollIntoView({ behavior: "smooth" });
      document.getElementById("followers_count").focus();
    });
  }

  if (btnScrollTop) {
    btnScrollTop.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
