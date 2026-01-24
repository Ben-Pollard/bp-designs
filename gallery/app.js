/**
 * BP Designs Experiment Gallery
 * Data-driven viewer for experiment results
 */

// Configuration
const EXPERIMENTS_PATH = '../output/experiments';

// State
let experiments = [];
let currentExperiment = null;

// DOM Elements
let experimentSelect;
let refreshBtn;
let experimentInfo;
let experimentTitle;
let patternType;
let totalVariants;
let successfulVariants;
let failedVariants;
let experimentDate;
let experimentDescription;
let variantsSection;
let variantsGrid;
let fixedParamsSection;
let fixedParamsGrid;
let emptyState;
let errorState;
let errorMessage;

/**
 * Initialize the gallery
 */
async function init() {
    // Get DOM elements
    experimentSelect = document.getElementById('experiment-select');
    refreshBtn = document.getElementById('refresh-btn');
    experimentInfo = document.getElementById('experiment-info');
    experimentTitle = document.getElementById('experiment-title');
    patternType = document.getElementById('pattern-type');
    totalVariants = document.getElementById('total-variants');
    successfulVariants = document.getElementById('successful-variants');
    failedVariants = document.getElementById('failed-variants');
    experimentDate = document.getElementById('experiment-date');
    experimentDescription = document.getElementById('experiment-description');
    variantsSection = document.getElementById('variants-section');
    variantsGrid = document.getElementById('variants-grid');
    fixedParamsSection = document.getElementById('fixed-params-section');
    fixedParamsGrid = document.getElementById('fixed-params-grid');
    emptyState = document.getElementById('empty-state');
    errorState = document.getElementById('error-state');
    errorMessage = document.getElementById('error-message');

    // Setup event listeners
    experimentSelect.addEventListener('change', onExperimentChange);
    refreshBtn.addEventListener('click', onRefresh);

    // Load experiments
    await loadExperiments();
}

/**
 * Load available experiments
 */
async function loadExperiments() {
    try {
        // Load experiments list from gallery directory
        const response = await fetch('./experiments.json');

        if (!response.ok) {
            // Index doesn't exist yet (no experiments run)
            experiments = [];
            showEmptyState();
            return;
        }

        const indexData = await response.json();

        // Extract experiment names from index
        experiments = indexData.map(entry => entry.name);

        if (experiments.length === 0) {
            showEmptyState();
        } else {
            hideEmptyState();
            renderExperimentList();
        }
    } catch (error) {
        console.error('Error loading experiments:', error);
        showError(`Failed to load experiments: ${error.message}`);
    }
}

/**
 * Render experiment list in dropdown
 */
function renderExperimentList() {
    experimentSelect.innerHTML = '<option value="">Select an experiment...</option>';

    experiments.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        experimentSelect.appendChild(option);
    });

    experimentSelect.disabled = false;
}

/**
 * Handle experiment selection change
 */
async function onExperimentChange(event) {
    const experimentName = event.target.value;

    if (!experimentName) {
        hideExperimentInfo();
        return;
    }

    await loadExperiment(experimentName);
}

/**
 * Load experiment data
 */
async function loadExperiment(name) {
    try {
        // Load config.json
        const configPath = `${EXPERIMENTS_PATH}/${name}/config.json`;
        const configResponse = await fetch(configPath);

        if (!configResponse.ok) {
            throw new Error(`Config not found: ${configPath}`);
        }

        const config = await configResponse.json();
        currentExperiment = { name, config };

        // Load variants
        const variants = await loadVariants(name, config);
        currentExperiment.variants = variants;

        // Render
        renderExperiment(currentExperiment);
    } catch (error) {
        console.error('Error loading experiment:', error);
        showError(`Failed to load experiment "${name}": ${error.message}`);
    }
}

/**
 * Load variant data for experiment
 */
async function loadVariants(experimentName, config) {
    const variants = [];
    const totalCount = config.total_variants || 0;

    // Load each variant's JSON
    for (let i = 1; i <= totalCount; i++) {
        const variantId = `var_${String(i).padStart(4, '0')}`;
        const variantPath = `${EXPERIMENTS_PATH}/${experimentName}/outputs/${variantId}.json`;

        try {
            const response = await fetch(variantPath);
            if (response.ok) {
                const variantData = await response.json();
                // Update SVG path to be relative to gallery
                variantData.svg_path = `${EXPERIMENTS_PATH}/${experimentName}/outputs/${variantId}.svg`;
                variants.push(variantData);
            }
        } catch (error) {
            console.warn(`Failed to load variant ${variantId}:`, error);
        }
    }

    return variants;
}

/**
 * Render experiment and variants
 */
function renderExperiment(experiment) {
    const { name, config, variants } = experiment;

    // Show experiment info
    experimentInfo.classList.remove('hidden');
    experimentTitle.textContent = config.experiment_name || name;
    patternType.textContent = config.pattern_type || '—';
    totalVariants.textContent = config.total_variants || 0;
    successfulVariants.textContent = config.successful || 0;
    failedVariants.textContent = config.failed || 0;

    if (config.timestamp) {
        const date = new Date(config.timestamp);
        experimentDate.textContent = date.toLocaleString();
    } else {
        experimentDate.textContent = '—';
    }

    if (config.description) {
        experimentDescription.textContent = config.description;
        experimentDescription.style.display = 'block';
    } else {
        experimentDescription.style.display = 'none';
    }

    // Render fixed parameters
    renderFixedParameters(config.parameters);

    // Render variants
    const variedParamNames = config.parameters && config.parameters.varied
        ? Object.keys(config.parameters.varied)
        : [];
    renderVariants(variants, variedParamNames);
}

/**
 * Render fixed parameters
 */
function renderFixedParameters(parameters) {
    fixedParamsGrid.innerHTML = '';

    if (!parameters || !parameters.fixed || Object.keys(parameters.fixed).length === 0) {
        fixedParamsSection.classList.add('hidden');
        return;
    }

    fixedParamsSection.classList.remove('hidden');

    Object.entries(parameters.fixed).forEach(([key, value]) => {
        const paramDiv = document.createElement('div');
        paramDiv.className = 'fixed-param';

        const paramName = document.createElement('span');
        paramName.className = 'param-name';
        paramName.textContent = key;

        const paramValue = document.createElement('span');
        paramValue.className = 'param-value';
        paramValue.textContent = formatValue(value);

        paramDiv.appendChild(paramName);
        paramDiv.appendChild(paramValue);
        fixedParamsGrid.appendChild(paramDiv);
    });
}

/**
 * Render variant grid
 */
function renderVariants(variants, variedParamNames) {
    variantsGrid.innerHTML = '';

    if (variants.length === 0) {
        variantsSection.classList.add('hidden');
        return;
    }

    variantsSection.classList.remove('hidden');

    variants.forEach(variant => {
        const card = createVariantCard(variant, variedParamNames);
        variantsGrid.appendChild(card);
    });
}

/**
 * Create variant card element
 */
function createVariantCard(variant, variedParamNames) {
    const card = document.createElement('div');
    card.className = 'variant-card';

    // Image container
    const imageContainer = document.createElement('div');
    imageContainer.className = 'variant-image';

    const img = document.createElement('img');
    img.src = variant.svg_path;
    img.alt = variant.variant_id;
    img.loading = 'lazy'; // Lazy load for performance
    imageContainer.appendChild(img);

    // Metadata container
    const metadata = document.createElement('div');
    metadata.className = 'variant-metadata';

    const variantId = document.createElement('div');
    variantId.className = 'variant-id';
    variantId.textContent = variant.variant_id;

    const params = document.createElement('div');
    params.className = 'variant-params';

    // Render parameters
    Object.entries(variant.params).forEach(([key, value]) => {
        // Only show parameters that were varied in the experiment
        if (variedParamNames.length > 0 && !variedParamNames.includes(key)) {
            return;
        }

        const paramDiv = document.createElement('div');
        paramDiv.className = 'variant-param';

        const paramName = document.createElement('span');
        paramName.className = 'param-name';
        paramName.textContent = key;

        const paramValue = document.createElement('span');
        paramValue.className = 'param-value';
        paramValue.textContent = formatValue(value);

        paramDiv.appendChild(paramName);
        paramDiv.appendChild(paramValue);
        params.appendChild(paramDiv);
    });

    metadata.appendChild(variantId);
    metadata.appendChild(params);

    card.appendChild(imageContainer);
    card.appendChild(metadata);

    return card;
}

/**
 * Format parameter value for display
 */
function formatValue(value) {
    if (typeof value === 'number') {
        // Format floats nicely
        return Number.isInteger(value) ? value : value.toFixed(4).replace(/\.?0+$/, '');
    }
    return String(value);
}

/**
 * Handle refresh button click
 */
async function onRefresh() {
    experimentSelect.value = '';
    experimentSelect.disabled = true;
    experimentSelect.innerHTML = '<option>Loading experiments...</option>';
    hideExperimentInfo();
    await loadExperiments();
}

/**
 * Show empty state
 */
function showEmptyState() {
    emptyState.classList.remove('hidden');
    errorState.classList.add('hidden');
    experimentInfo.classList.add('hidden');
    variantsSection.classList.add('hidden');
}

/**
 * Hide empty state
 */
function hideEmptyState() {
    emptyState.classList.add('hidden');
    errorState.classList.add('hidden');
}

/**
 * Hide experiment info
 */
function hideExperimentInfo() {
    experimentInfo.classList.add('hidden');
    variantsSection.classList.add('hidden');
}

/**
 * Show error state
 */
function showError(message) {
    errorState.classList.remove('hidden');
    emptyState.classList.add('hidden');
    experimentInfo.classList.add('hidden');
    variantsSection.classList.add('hidden');
    errorMessage.textContent = message;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
