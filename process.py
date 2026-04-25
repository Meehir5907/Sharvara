import os
import bz2
import socket
import pandas as pd
import dpkt
import torch
from tqdm import tqdm

def load_ground_truth(binetflow_path, verbose=True):
    if verbose:
        print(f"\n[INFO] Loading ground truth from: {binetflow_path}")
        
    flow_df = pd.read_csv(binetflow_path)
    flow_targets = {}
    
    iterator = tqdm(flow_df.iterrows(), total=len(flow_df), desc="Parsing Binetflow", unit="rows", disable=not verbose)
    
    for _, row in iterator:
        src_ip = str(row['SrcAddr'])
        dst_ip = str(row['DstAddr'])
        src_port = str(row['Sport'])
        dst_port = str(row['Dport'])
        proto = str(row['Proto']).lower()
        label_val = str(row['Label'])
        
        tuple_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
        
        if tuple_key in flow_targets:
            if "Botnet" in label_val or "Normal" in label_val:
                flow_targets[tuple_key]['label'] = label_val
        else:
            flow_targets[tuple_key] = {
                'label': label_val,
                'duration': row['Dur'],
                'tot_bytes': row['TotBytes'],
                'tot_pkts': row['TotPkts'],
                'state': row['State']
            }
            
    if verbose:
        print(f"[SUCCESS] Mapped {len(flow_targets)} unique connections.\n")
        
    return flow_targets

def mask_packet_data(raw_buffer):
    buf_array = bytearray(raw_buffer)
    buf_len = len(buf_array)
    
    # Mask MAC addresses
    for idx in range(min(12, buf_len)):
        buf_array[idx] = 0
        
    # Mask Source/Destination IPs
    if buf_len >= 34:
        for idx in range(26, 34):
            buf_array[idx] = 0
            
    return bytes(buf_array)

def extract_packet_features(raw_buffer, prev_time, current_time):
    time_delta = current_time - prev_time if prev_time else 0.0
    masked_buffer = mask_packet_data(raw_buffer)
    
    header_bytes = list(masked_buffer[:60])
    if len(header_bytes) < 60:
        header_bytes.extend([0] * (60 - len(header_bytes)))
        
    return time_delta, header_bytes

def build_dataset(pcap_path, truth_dict, max_packets=10, out_dir="tensors", verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    active_flows = {}
    flow_prev_time = {}
    
    saved_stats = {"botnet": 0, "normal": 0, "bg": 0}
    limit_botnet = 40000
    limit_normal = 30000
    limit_bg = 40000
    
    if verbose:
        print(f"[INFO] Initializing PCAP stream from: {pcap_path}")
        print(f"[INFO] Target limits -> Botnet: {limit_botnet} | Normal: {limit_normal} | BG: {limit_bg}")
        print(f"[INFO] Output directory: ./{out_dir}/")
        
    pbar = tqdm(total=(limit_botnet + limit_normal + limit_bg), desc="Tensors Generated", unit="tensors", disable=not verbose)
    
    with bz2.open(pcap_path, 'rb') as pcap_file:
        pcap_reader = dpkt.pcapng.Reader(pcap_file)
        
        # Manual iteration to survive any dpkt parser crashes on malformed packets
        while True:
            try:
                timestamp, buf = next(pcap_reader)
            except StopIteration:
                break
            except Exception:
                continue
                
            try:
                eth_frame = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth_frame.data, dpkt.ip.IP):
                    continue
                
                ip_packet = eth_frame.data
                trans_layer = ip_packet.data
                
                if not isinstance(trans_layer, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    continue
                    
                src_ip = socket.inet_ntoa(ip_packet.src)
                dst_ip = socket.inet_ntoa(ip_packet.dst)
                src_port = str(trans_layer.sport)
                dst_port = str(trans_layer.dport)
                proto = 'tcp' if isinstance(trans_layer, dpkt.tcp.TCP) else 'udp'
                
                # Check BOTH directions of the flow to capture full bidirectional behavior
                fwd_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
                rev_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{proto}"
                
                if fwd_key in truth_dict:
                    tuple_key = fwd_key
                elif rev_key in truth_dict:
                    tuple_key = rev_key
                else:
                    continue
                    
                lbl = str(truth_dict[tuple_key]['label'])
                
                # Robust substring matching
                is_botnet = "Botnet" in lbl
                is_normal = "Normal" in lbl
                is_bg = not is_botnet and not is_normal
                
                if is_botnet and saved_stats["botnet"] >= limit_botnet:
                    continue
                elif is_normal and saved_stats["normal"] >= limit_normal:
                    continue
                elif is_bg and saved_stats["bg"] >= limit_bg:
                    continue
                
                if tuple_key not in active_flows:
                    active_flows[tuple_key] = []
                    flow_prev_time[tuple_key] = timestamp
                    
                prev_t = flow_prev_time[tuple_key]
                t_delta, h_bytes = extract_packet_features(buf, prev_t, timestamp)
                
                active_flows[tuple_key].append({
                    'delta': t_delta,
                    'bytes': h_bytes
                })
                flow_prev_time[tuple_key] = timestamp
                
                if len(active_flows[tuple_key]) == max_packets:
                    stat_data = truth_dict[tuple_key]
                    flow_tensor = torch.tensor([pkt['bytes'] for pkt in active_flows[tuple_key]], dtype=torch.float32)
                    
                    tensor_idx = saved_stats["botnet"] + saved_stats["normal"] + saved_stats["bg"]
                    safe_filename = tuple_key.replace(':', '_').replace('/', '_') + f"_{tensor_idx}"
                    save_path = os.path.join(out_dir, f"{safe_filename}.pt")
                    
                    torch.save({
                        'tensor': flow_tensor, 
                        'stats': stat_data
                    }, save_path)
                    
                    if is_botnet:
                        saved_stats["botnet"] += 1
                    elif is_normal:
                        saved_stats["normal"] += 1
                    else:
                        saved_stats["bg"] += 1
                    
                    del active_flows[tuple_key]
                    pbar.update(1)
                    
                    if saved_stats["botnet"] >= limit_botnet and saved_stats["normal"] >= limit_normal and saved_stats["bg"] >= limit_bg:
                        break
                        
            except Exception:
                continue
                
    pbar.close()
    if verbose:
        print("\n[SUCCESS] Dataset extraction complete.")
        print(f"Botnet Tensors: {saved_stats['botnet']}")
        print(f"Normal Tensors: {saved_stats['normal']}")
        print(f"Background Tensors: {saved_stats['bg']}")

if __name__ == "__main__":
    truth_data = load_ground_truth("datasetCTU/b42/capture20110810.binetflow.2format", verbose=True)
    build_dataset("datasetCTU/b42/capture20110810.truncated.pcap.bz2", truth_data, verbose=True)
