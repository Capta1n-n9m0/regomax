// noinspection DuplicatedCode

const fsp = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");
const Matrix = require("./Matrix");
const printf = require("printf");
const Vector = require("./Vector");
const os = require("os");
const nCpu = os.cpus().length;
const { Worker } = require("worker_threads");

const eps_pagerank = 1e-13;

/**
 * @return {number}
 */
function getTime() {
    return Number(process.hrtime.bigint() / BigInt(1_000_000));
}

/**
 * @param{Vector} pagerank
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{number} iprint
 * @param{Vector} node
 * @param{number} trans_frag
 * @return {Promise<number>}
 */
async function calc_pg_proj_th(pagerank, net, delta_alpha, iprint, node, trans_frag = 0){
    return new Promise((resolve)=>{
        let w = new Worker(path.join(__dirname, "Thread.js"));
        w.on("message", (msg)=>{
            console.log(`calculations took ${msg.delay}ms`);
            resolve(msg.data);
        });
        w.postMessage({
            options:{
                work: true,
                task: 1,
                once: true,
            },
            data: {pagerank, net, delta_alpha, iprint, node, trans_frag}
        });
    });
}

/**
 * @param{Vector} right
 * @param{Vector} left
 * @param{Vector} pg
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{Vector} node
 * @return Promise<number>
 */
async function compute_project(right, left, pg, net, delta_alpha, node) {
    const iprint = 10;
    console.log("compute_project()");
    let sp, dlambda1, dlambda2, dlambda3;
    let node0 = new Vector(0);

    right.put_value(1.0);
    left.put_value(1.0);
    pg.put_value(1.0);

// #pragma omp parallel sections
    {
// #pragma omp section
        let p2 = calc_pg_proj_th(left, net, delta_alpha, iprint, node, 1);
// #pragma omp section
        let p1 = calc_pg_proj_th(right, net, delta_alpha, iprint, node);
// #pragma omp section
        let p3 = calc_pg_proj_th(pg, net, delta_alpha, iprint, node0);

        dlambda1 = await p1;
        dlambda2 = await p2;
        dlambda3 = await p3;
    }

    sp = 1.0 / Vector.scalar_product(left, right);
    left.mul_eq(sp);
    sp = Vector.scalar_product(left, right);
// #pragma omp critical(print)
    {
        console.log(printf("dlambda = %24.16f   diff = %f\n",
            dlambda1, Math.abs(dlambda1 - dlambda2)));
        console.log(printf("TEST: psi_left^T * psi_right = %26.16f\n", sp));
        //     fflush(stdout);
    }
    return dlambda1;
}

/**
 * @type {Worker[]}
 */
let threads = new Array(nCpu);

async function send_data(data){
    console.log("send_data()");
    return new Promise((resolve)=>{
        let done_counter = 0;
        for(let i = 0; i < threads.length; i++){
            threads[i].executor = (msg)=>{
                console.log(`${msg.id} ready in ${msg.delay}ms`);
                done_counter++;
                if(done_counter === threads.length){
                    for(let j = 0; j < threads[j]; j++){
                        threads[j].off("message", threads[j].executor);
                    }
                    resolve(1);
                }
            }
            threads[i].on("message", threads[i].executor);
        }
        for (let i = 0; i < threads.length; i++) {
            threads[i].postMessage({data, options:{work: true, once: false, stage: 1, id: i, task: 2}})
        }
    })
}

async function compute_GR_heavy(data){
    console.log("compute_GR_heavy()");
    return new Promise(async (resolve) => {
        let nr = data.node.dim;
        for (let i = 0; i < nCpu; i++) {
            threads[i] = new Worker(path.join(__dirname, "Thread.js"));
            // threads[i].postMessage({data, options:{work: true, once: false, stage: 1, id: i}});
            // threads[i].on("message", setup(i, threads));
        }
        send_data(data).then(()=>{
            console.log("called then!");
            let c = -1;
            let done_counter = 0;
            for(let i = 0; i < threads.length; i++){
                threads[i].executor = (msg)=>{
                    console.log(`${msg.id} ready in ${msg.delay}ms`);
                    c++;
                    done_counter++;
                    if(c < nr){
                        threads[i].postMessage({data:{i: c}, options:{work:true, once: false, stage: 2, task: 2}})
                    }
                    if(done_counter === nr){
                        for(let j = 0; j < threads.length; j++){
                            threads[j].postMessage({options: {work: false}});
                            threads[j].off("message", threads[j].executor);
                        }
                        resolve();
                    }
                }
                threads[i].on("message", threads[i].executor);
            }
            for(let i = 0; i < threads.length; i++){
                c++;
                if(c < nr){
                    threads[i].postMessage({data:{i: c}, options:{work: true, once: false, stage: 2, task: 2}});
                }
            }
        });

    })
}

/**
 * @param{Matrix} G_R
 * @param{Matrix} G_rr
 * @param{Matrix} G_pr
 * @param{Matrix} G_qr
 * @param{Matrix} G_I
 * @param{Vector} psiL
 * @param{Vector} psiR
 * @param{Vector} pg
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{Vector} node
 */
async function compute_GR(G_R, G_rr, G_pr,
                          G_qr, G_I, psiL,
                          psiR, pg, net,
                          delta_alpha, node) {
    console.log("compute_GR()");
    let nr = node.dim;
    if (G_R.x !== nr || G_R.y !== nr) throw "Wrong matrix size of G_R  in compute_GR";
    if (G_rr.x !== nr || G_rr.y !== nr) throw "Wrong matrix size of G_rr  in compute_GR";
    if (G_pr.x !== nr || G_pr.y !== nr) throw "Wrong matrix size of G_pr  in compute_GR";
    if (G_qr.x !== nr || G_qr.y !== nr) throw "Wrong matrix size of G_qr  in compute_GR";
    if (G_I.x !== nr || G_I.y !== nr) throw "Wrong matrix size of G_I  in compute_GR";
    let dlambda;

    let max_iter;

    max_iter = Math.floor(-Math.log(eps_pagerank) / (delta_alpha + 3e-7));
    max_iter *= 2;

    console.log("Computation of left and right eigenvectors of G_ss");
    dlambda = await compute_project(psiR, psiL, pg, net, delta_alpha, node);

    // note that the last line also fixes the default size of dvec to n
    // which is important in the private declaration below which implicitely
    // calls the default constructor of dvec for each thread

    // #pragma omp parallel for schedule(dynamic) private(in, out, s, t, f, f2, j, l, quality)
    const data = {G_R, G_rr, G_pr, G_qr, G_I, psiL, psiR, pg, net, delta_alpha, node, max_iter, dlambda};
    await compute_GR_heavy(data);
}

async function main(argv) {
    const start = getTime();
    const netfile = argv[2];
    const delta_alpha = parseFloat(argv[3]);
    const iprint = parseInt(argv[4]);
    const print_number = parseInt(argv[5]);
    const ten_number = parseInt(argv[6]);
    let nodefile = argv[7];
    const nodefilenames = argv[8];
    console.log(printf("input-file, 1-alpha, iprint, print_number, ten_number  = %s  %f  %d  %d  %d\n", netfile, delta_alpha, iprint, print_number, ten_number));
    console.log(printf("nodefile = %s\n", nodefile));
    console.log(printf("file of node names = %s\n", nodefilenames));

    let node;
    const len = await fsp.readFile(nodefile)
        .then(data => {
            const lines = bsplit(data, Buffer.from("\n"));
            let len = parseInt(lines[0].toString());
            node = new Vector(len);
            let i = 0;
            for (const line of lines.slice(1)) {
                if (line.length === 0) continue;
                node.c[i++] = parseInt(line.toString());
            }
            return len;
        });
    console.log(printf("reading of nodefile finished: len = %d\n", len));
    const net = new Network(netfile);
    let GR = new Matrix(len, len),
        Grr = new Matrix(len, len),
        Gpr = new Matrix(len, len),
        Gqr = new Matrix(len, len),
        GI = new Matrix(len, len);
    const n = net.size;
    let psiL = new Vector(n),
        psiR = new Vector(n),
        pg = new Vector(n);
    nodefile = path.basename(nodefile, ".nodes");
    await compute_GR(GR, Grr, Gpr, Gqr, GI, psiL, psiR, pg, net, delta_alpha, node);
    Matrix.print_mat(Gqr, `Gqr_${net.base_name}_${nodefile}_${len}.dat`, nodefilenames);
    console.log(`Calculations took ${(getTime() - start) / 1000} sec\n`);
}

module.exports = main;
