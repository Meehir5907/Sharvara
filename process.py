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
    flow_dict = {}
    
    # Wrap the iterrows in a tqdm progress bar
    iterator = tqdm(
        flow_df.iterrows(), 
        total=len(flow_df), 
        desc="Parsing Binetflow", 
        unit="rows",
        disable=not verbose
    )
    
    for _, row in iterator:
        src_ip = str(row['SrcAddr'])
        dst_ip = str(row['DstAddr'])
        src_port = str(row['Sport'])
        dst_port = str(row['Dport'])
        proto = str(row['Proto']).lower()
        
        tuple_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
        
        flow_dict[tuple_key] = {
            'label': row['Label'],
            'duration': row['Dur'],
            'tot_bytes': row['TotBytes'],
            'tot_pkts': row['TotPkts'],
            'state': row['State']
        }
        
    if verbose:
        print(f"[SUCCESS] Mapped {len(flow_dict)} unique 5-tuple connections.\n")
        
    return flow_dict

def extract_packet_features(raw_buffer, prev_time, current_time):
    time_delta = current_time - prev_time if prev_time else 0.0
    
    header_bytes = list(raw_buffer[:60])
    if len(header_bytes) < 60:
        header_bytes.extend([0] * (60 - len(header_bytes)))
        
    return time_delta, header_bytes

def build_dataset(pcap_path, truth_dict, max_packets=10, out_dir="tensors", verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    active_flows = {}
    flow_prev_time = {}
    
    if verbose:
        print(f"[INFO] Initializing PCAP stream from: {pcap_path}")
        print(f"[INFO] Target tensor dimension: ({max_packets} packets x 60 bytes)")
        print(f"[INFO] Output directory: ./{out_dir}/")
        
    # We track progress based on how many tensors we successfully save
    pbar = tqdm(desc="Tensors Generated", unit="tensors", disable=not verbose)
    
    with bz2.open(pcap_path, 'rb') as pcap_file:
        pcap_reader = dpkt.pcapng.Reader(pcap_file)
        
        for timestamp, buf in pcap_reader:
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
                
                tuple_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
                
                if tuple_key in truth_dict:
                    if tuple_key not in active_flows:
                        active_flows[tuple_key] = []
                        flow_prev_time[tuple_key] = timestamp
                        
                    if len(active_flows[tuple_key]) < max_packets:
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
                            
                            safe_filename = tuple_key.replace(':', '_').replace('/', '_')
                            save_path = os.path.join(out_dir, f"{safe_filename}.pt")
                            
                            torch.save({
                                'tensor': flow_tensor, 
                                'stats': stat_data
                            }, save_path)
                            
                            del active_flows[tuple_key]
                            pbar.update(1)
                            
            except Exception:
                continue
                
    pbar.close()
    if verbose:
        print("\n[SUCCESS] Dataset extraction complete.")

if __name__ == "__main__":
    truth_data = load_ground_truth("datasetCTU/b42/capture20110810.binetflow.2format", verbose=True)
    build_dataset("datasetCTU/b42/capture20110810.truncated.pcap.bz2", truth_data, verbose=True)
