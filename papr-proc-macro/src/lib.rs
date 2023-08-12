use proc_macro::TokenStream;
use quote::quote;
use syn::{
    braced,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    token::Comma,
    Ident, ItemStruct, Token,
};

struct NodeConstructorParser {
    struc: ItemStruct,
    inputs: Punctuated<Ident, Comma>,
    outputs: Punctuated<Ident, Comma>,
}

impl Parse for NodeConstructorParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let struc: ItemStruct = input.parse()?;

        input.parse::<Token![in]>()?;
        let inputs;
        braced!(inputs in input);
        let inputs = inputs.parse_terminated(Ident::parse, Token![,])?;

        match input.parse::<Ident>()?.to_string().as_str() {
            "out" => {}
            _ => return Err(input.error("expected `out`")),
        };
        let outputs;
        braced!(outputs in input);
        let outputs = outputs.parse_terminated(Ident::parse, Token![,])?;

        Ok(Self {
            struc,
            inputs,
            outputs,
        })
    }
}

#[proc_macro]
pub fn node_constructor(tokens: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(tokens as NodeConstructorParser);

    let NodeConstructorParser {
        struc,
        mut inputs,
        outputs,
    } = parsed;

    let struc_name = &struc.ident;

    let inputs_args = inputs
        .iter()
        .map(|inp| quote! { #inp: crate::Scalar })
        .collect::<Punctuated<_, Comma>>();

    let fields_args = &struc
        .fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            quote! { #ident: #ty }
        })
        .collect::<Punctuated<_, Comma>>();

    let fields = &struc
        .fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            quote! { #ident }
        })
        .collect::<Punctuated<_, Comma>>();

    inputs.push(syn::parse2::<Ident>(quote! { t }).unwrap());

    let outs = outputs
        .iter()
        .map(|out| {
            quote! {
                crate::graph::Output { name: stringify!(#out).to_owned() }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let ins = inputs
        .iter()
        .map(|inp| {
            quote! {
                crate::graph::Input::new(stringify!(#inp), Some(crate::dsp::Signal::new(0.0)))
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let args = if fields_args.is_empty() {
        quote! { #inputs_args }
    } else {
        quote! { #fields_args, #inputs_args }
    };

    let input_names_list = inputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let output_names_list = outputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let input_mappings = inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    let output_mappings = outputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    quote! {
        #[derive(Clone)]
        #struc

        #[allow(unused_variables)]
        impl #struc_name {
            pub const INPUTS: &[&'static str] = &[#input_names_list];
            pub const OUTPUTS: &[&'static str] = &[#output_names_list];

            pub fn input_idx(name: &str) -> Option<usize> {
                match name {
                    #input_mappings
                    _ => None,
                }
            }

            pub fn output_idx(name: &str) -> Option<usize> {
                match name {
                    #output_mappings
                    _ => None,
                }
            }


            #[allow(clippy::too_many_arguments)]
            pub fn create_node(name: &str, signal_rate: crate::dsp::SignalRate, buffer_len: usize, #args) -> std::sync::Arc<crate::graph::Node> {
                let this = Box::new(std::sync::RwLock::new(Self { #fields }));
                let an = std::sync::Arc::new(crate::graph::Node::new(
                    crate::graph::NodeName::new(name),
                    signal_rate,
                    buffer_len,
                    vec![#ins],
                    vec![#outs],
                    crate::graph::ProcessorType::Builtin(this),
                ));
                an
            }
        }
    }.into()
}
