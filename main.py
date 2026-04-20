"""
main.py
Orchestratore principale per il TomatoProject v2.

Collega tutti i moduli (Dati, Modello, Training, Valutazione, Plotting) 
per eseguire l'esperimento completo di classificazione dello stress nelle piante 
usando Spiking Neural Networks (SNN) ed E-prop/BPTT.

Autore: Shanti Leonardo Arzu
Data: Aprile 2026
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import dei nostri moduli
from src.data_processing_manager import PlantDataManager
from src.snn_layer_model import RecurrentLayer, FeedforwardLayer, compute_decay_factors
from src.learning_trainer import SNNTrainer
from src.learning_evaluator import ModelEvaluator
from src import plot_visualizations_new as plots


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ PARAMETRI ESPERIMENTO / DATI                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
STRESS_TYPE = "water"
LEAVE_PLANT = "P3"
RANDOM_SPLIT = True
SEED = 42


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ PARAMETRI DI TEMPORAL ENCODING                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
ENCODING_TYPE = "rate"
NB_STEPS = 150
DT = 1.0
GAIN_RATE = 10.0
GAIN_LIF = 0.35
INPUT_SHIFT = 3.8
POPULATION_SIZE = 1

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ IPERPARAMETRI ENCODER LIF                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
TAU_MEM = 18.0
TAU_SYN = 12.0
TAU_REF_ENCODER = 3.5
NOISE_STD = 1.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ IPERPARAMETRI RETE RICORRENTE (SNN - HIDDEN LAYER)                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
HIDDEN_NEURONS = 300
TAU_MEM_REC = 35.0
TAU_REF = 2.5
THRESHOLD = 0.80
TAU_TRACE = 80.0
TAU_TRACE_OUT = 105.0
W_IN_SCALE = 0.90
W_REC_SCALE = 0.40
W_OUT_SCALE = 0.90


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ IPERPARAMETRI TRAINING                                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
ALGORITHM = "eprop" # "eprop" o "bptt"
EPOCHS = 100
BATCH_SIZE = 24
LEARNING_RATE = 0.003
GAMMA = 0.3


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FLAG OPERATIVI                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
GENERATE_PLOTS = True          # True: genera grafici | False: disabilita plotting
SAVE_MODEL = True              # True: salva modello addestrato


def print_configuration():
    """Stampa una chiara tabella riassuntiva dei parametri."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 75)
    print(f"TOMATO PROJECT V2 - SNN {ALGORITHM.upper()} ({STRESS_TYPE.upper()} STRESS)")
    print("=" * 75)
    print(f" DEVICE & SEED           | Device: {device} | Seed: {SEED}")
    print("-" * 75)
    print(f" DATASET SPLIT           | Mode: {'Random 70/30' if RANDOM_SPLIT else f'LOPO (Test: {LEAVE_PLANT})'}")
    print("-" * 75)
    print(f" TEMPORAL ENCODING (LIF) | Type: {ENCODING_TYPE.upper()} | Steps: {NB_STEPS} | dt: {DT}ms")
    print(f"                         | Gain_lif: {GAIN_LIF} | τ_syn: {TAU_SYN}ms | τ_mem: {TAU_MEM}ms")
    print(f"                         | τ_ref_encoder: {TAU_REF_ENCODER}ms | Noise: {NOISE_STD} [EQUILIBRIO] | Input_shift: {INPUT_SHIFT}")
    print(f"                         | Population_size: {POPULATION_SIZE} → nb_inputs = 6 × {POPULATION_SIZE} = {6*POPULATION_SIZE}")
    print("-" * 75)
    print(f" SNN ARCHITECTURE        | Hidden: {HIDDEN_NEURONS} | Threshold: {THRESHOLD}")
    print(f"                         | τ_mem_rec: {TAU_MEM_REC}ms | τ_ref: {TAU_REF}ms")
    print(f"                         | τ_trace: {TAU_TRACE}ms | τ_trace_out: {TAU_TRACE_OUT}ms")
    print(f"                         | Weight scales: W_in={W_IN_SCALE} W_rec={W_REC_SCALE} W_out={W_OUT_SCALE}")
    print("-" * 75)
    print(f" TRAINING CONFIGURATION  | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"                         | Gamma: {GAMMA} | Algorithm: {ALGORITHM.upper()}")
    print("=" * 75 + "\n")


def main():
    """Esegue l'esperimento completo di training e valutazione."""
    
    # Stampa configurazione
    print_configuration()
    
    # Setup device
    if SEED > 0:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    
    #per mac os con gpu m1/m2
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    # per server con cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"✓ Device: {device}")
    print(f"   DEBUG: ENCODING_TYPE = '{ENCODING_TYPE}' \n")

    # Load dataset
    print("[1/5] Caricamento e split Dataset...")
    
    encoding_params = {
        "encoding_type": ENCODING_TYPE,
        "nb_steps": NB_STEPS,
        "dt": DT,
        "gain_rate": GAIN_RATE,
        "gain_lif": GAIN_LIF,
        "tau_syn": TAU_SYN,
        "tau_mem": TAU_MEM,
        "threshold": THRESHOLD,
        "input_shift": INPUT_SHIFT,
        "tau_ref": TAU_REF_ENCODER,
        "noise_std": NOISE_STD,
        "population_size": POPULATION_SIZE,
    }
    
    data_manager = PlantDataManager(stress_type=STRESS_TYPE.lower(), encoding_params=encoding_params)
    file_path = f"./data/{STRESS_TYPE.lower()}_stress/{STRESS_TYPE.capitalize()}_Stress.npz"
    
    if not RANDOM_SPLIT:
        result = data_manager.prepare_dataset_leave_one_plant_split(
            file_path=file_path, leave_plant=LEAVE_PLANT, val_size=0.0
        )
        ds_train, ds_test, metadata = result[0], result[1], result[2]
        print(f"✓ Modalità LOPO: pianta di test = {LEAVE_PLANT}")
    else:
        result = data_manager.prepare_dataset_standard_split(
            file_path=file_path, test_size=0.3, val_size=0.0
        )
        ds_train, ds_test, metadata = result[0], result[1], result[2]
        print(f"✓ Modalità Random Split: 70% train / 30% test")
    
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    nb_inputs = metadata.get("nb_inputs", 6) * POPULATION_SIZE
    nb_outputs = metadata.get("nb_outputs", 3)
    max_time_ms = int(NB_STEPS * DT)
    print(f"✓ Dati caricati: {len(ds_train)} train | {len(ds_test)} test\n")

    # Initialize model
    print("[2/5] Inizializzazione Rete Spiking (LIF)...")
    w_in, w_rec = RecurrentLayer.create_layer(
        nb_inputs=nb_inputs, nb_outputs=HIDDEN_NEURONS, fwd_scale=W_IN_SCALE, rec_scale=W_REC_SCALE, device=device
    )
    w_out = FeedforwardLayer.create_layer(
        nb_inputs=HIDDEN_NEURONS, nb_outputs=nb_outputs, scale=W_OUT_SCALE, device=device
    )
    layers = (w_in, w_out, w_rec)

    decay_factors = compute_decay_factors(
        dt=DT, 
        tau_mem=TAU_MEM,          # Encoder membrana (encoding layer)
        tau_mem_rec=TAU_MEM_REC,  # Rete ricorrente membrana
        tau_syn=TAU_SYN,          # Encoder sinaptico
        tau_trace=TAU_TRACE, 
        tau_trace_out=TAU_TRACE_OUT
    )

    run_snn_kwargs = {
        "decay": decay_factors,
        "nb_steps": NB_STEPS,
        "nb_hidden": HIDDEN_NEURONS,
        "nb_outputs": nb_outputs,
        "device": device,
        "lower_bound": -1.0,
        "ref_per_timesteps": int(TAU_REF / DT) if TAU_REF > 0 else 0,
        "tau_mem_ms": TAU_MEM,
        "max_time_ms": max_time_ms,
        "gamma": GAMMA,
        "threshold": THRESHOLD
    }
    print(f"✓ Modello creato: {nb_inputs}→{HIDDEN_NEURONS}→{nb_outputs}\n")

    # Training
    print(f"[3/5] Addestramento con {ALGORITHM.upper()}...")
    trainer = SNNTrainer(
        layers=layers, 
        device=device, 
        nb_outputs=nb_outputs, 
        tau_mem_ms=TAU_MEM, 
        max_time_ms=max_time_ms, 
        lr=LEARNING_RATE, 
        algorithm=ALGORITHM,
        # ─── PARAMETRI SCHEDULER ───
        scheduler_patience=5,      # Riduci dopo 5 epoche senza miglioramento
        scheduler_factor=0.5,      # Riduci LR del 50%
        scheduler_min_lr=1e-7      # Non scendere sotto 1e-7
    )
    
    history, weight_history = trainer.fit(
        train_loader=train_loader, 
        epochs=EPOCHS, 
        run_snn_kwargs=run_snn_kwargs, 
        test_loader=test_loader
    )
    
    print(f"✓ Training completato\n")

    # Save model
    if SAVE_MODEL:
        os.makedirs("./results/models", exist_ok=True)
        model_name = f"{STRESS_TYPE}_{ENCODING_TYPE}_ep{EPOCHS}_hid{HIDDEN_NEURONS}.pt"
        model_path = os.path.join("./results/models", model_name)
        torch.save(layers, model_path)
        print(f"✓ Modello salvato in: {model_path}\n")

    # Evaluate
    print("[4/5] Estrazione Predizioni e Attività Neurale...")
    evaluator = ModelEvaluator(layers=layers, device=device, nb_outputs=nb_outputs)
    y_true, y_pred, spk_input, spk_hidden, spk_out = evaluator.collect_predictions_and_activity(
        test_loader=test_loader,
        run_snn_kwargs=run_snn_kwargs
    )
    print(f"✓ Valutazione completata\n")

    # Generate plots
    if GENERATE_PLOTS:
        print("[5/5] Generazione Grafici...")
        
        # Crea cartelle per salvare i risultati (struttura flat, 1 livello)
        os.makedirs("./results/training", exist_ok=True)
        os.makedirs("./results/spike_analysis", exist_ok=True)
        
        model_name = f"{STRESS_TYPE}_{ENCODING_TYPE}_{ALGORITHM}_ep{EPOCHS}_hid{HIDDEN_NEURONS}"
        
        class_labels = ["Control", "Early_Stress", "Late_Stress"]
        
        # Training metrics
        plots.plot_training_performance(
            history["train_acc"], 
            history.get("test_acc", []), 
            history["loss"], 
            f"./results/training/{model_name}_training.pdf"
        )
        plots.plot_confusion_matrix(
            y_true, y_pred, class_labels, 
            f"./results/training/{model_name}_confusion.pdf"
        )
        
        # Weight evolution
        plots.plot_weights_evolution(
            weight_history, 
            f"./results/training/{model_name}_weights_mean.pdf"
        )
        plots.plot_individual_weights_evolution(
            weight_history, 
            f"./results/training/{model_name}_weights_indiv.pdf", 
            num_weights_to_plot=15
        )
        
        # Spike analysis
        if len(spk_input) > 0:
            try:
                spr_recs = [spk_input[0][0], spk_hidden[0][0], spk_out[0][0]]
                layer_names = ["Encoding Input", "Hidden Layer", "Output Layer"]
                
                gain = GAIN_LIF if "lif" in ENCODING_TYPE else GAIN_RATE
                plots.plot_network_activity(
                    spr_recs, 
                    layer_names, 
                    f"./results/spike_analysis/{model_name}_raster", 
                    time_step=DT,
                    gain_factor=gain,
                    encoding_type=ENCODING_TYPE,
                    algorithm=ALGORITHM
                )
            except Exception as e:
                print(f"⚠️  Errore nella generazione del raster plot: {type(e).__name__}: {e}")
                print(f"   Dettagli:")
                print(f"   - spk_input[0] shape: {spk_input[0].shape}")
                print(f"   - spk_hidden[0] shape: {spk_hidden[0].shape}")
                print(f"   - spk_out[0] shape: {spk_out[0].shape}")
                print(f"   - Encoding type: {ENCODING_TYPE}")
                print(f"   - Algorithm: {ALGORITHM}")
        
        print(f"✓ Grafici salvati in:")
        print(f"   - Training (metrics, confusion, weights): ./results/training/")
        print(f"   - Spike analysis (raster plots): ./results/spike_analysis/\n")

    # Final report
    print("=" * 75)
    print("✨ ESPERIMENTO COMPLETATO CON SUCCESSO! ✨")
    print("=" * 75 + "\n")
    
    # Final metrics
    final_train_acc = history["train_acc"][-1]
    final_test_acc = history.get("test_acc", [])[-1] if history.get("test_acc") else 0.0
    final_loss = history["loss"][-1]
    
    print(f"📊 RISULTATI FINALI:")
    print(f"   Train Accuracy: {final_train_acc:.4f}")
    print(f"   Test Accuracy:  {final_test_acc:.4f}")
    print(f"   Loss Finale:    {final_loss:.4f}\n")


if __name__ == "__main__":
    main()